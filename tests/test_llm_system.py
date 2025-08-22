"""
Comprehensive test suite for new LLM trading system components.
Tests schemas, ATR sizing, liquidity checks, correlation, EV, and calibration.
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from robo_trader.schemas import (
    TradingDecision, TradingMode, Direction, TradeRecommendation,
    ComplianceCheck, RiskState, MarketData, NewsEvent
)
from robo_trader.llm_client import DecisiveLLMClient
from robo_trader.risk import RiskManager, Position
from robo_trader.market_meta import MarketMetadata, MarketMetaProvider
from robo_trader.correlation import CorrelationBudget, Sector
from robo_trader.edge import EdgeCalculator, EVCalculation
from robo_trader.calibration import CalibrationTracker


class TestSchemas:
    """Test trading decision schemas."""
    
    def test_trading_decision_validation(self):
        """Test that trading decisions validate correctly."""
        # Valid trade decision
        decision = TradingDecision(
            mode=TradingMode.TRADE,
            universe_checked=["AAPL", "GOOGL"],
            conviction=75,
            recommendation=TradeRecommendation(
                symbol="AAPL",
                direction=Direction.LONG,
                entry_type="limit",
                entry_price=150.0,
                position_size_bps=30,
                stop_loss=145.0,
                time_stop_hours=48,
                targets=[155.0, 160.0],
                thesis="Breakout above resistance",
                risk_reward=2.0,
                p_win=0.55,
                expected_value_pct=1.5
            ),
            compliance_checks=ComplianceCheck(
                liquidity_ok=True,
                spread_ok=True,
                borrow_ok=True,
                correlation_ok=True
            ),
            risk_state=RiskState(
                day_dd_bps=50,
                week_dd_bps=150,
                cash_pct=70.0
            )
        )
        
        assert decision.should_execute() is True
        assert "LONG AAPL" in decision.get_action_summary()
    
    def test_conviction_threshold(self):
        """Test conviction threshold for execution."""
        decision = TradingDecision(
            mode=TradingMode.TRADE,
            universe_checked=["AAPL"],
            conviction=45,  # Below threshold
            compliance_checks=ComplianceCheck(
                liquidity_ok=True,
                spread_ok=True,
                borrow_ok=True,
                correlation_ok=True
            ),
            risk_state=RiskState(day_dd_bps=0, week_dd_bps=0, cash_pct=100)
        )
        
        assert decision.should_execute() is False
    
    def test_risk_reward_validation(self):
        """Test risk:reward ratio validation."""
        with pytest.raises(ValueError, match="Risk:reward must be >= 1.8"):
            TradeRecommendation(
                symbol="AAPL",
                direction=Direction.LONG,
                entry_type="market",
                entry_price=150.0,
                position_size_bps=30,
                stop_loss=148.0,
                time_stop_hours=24,
                targets=[151.0],
                thesis="Test",
                risk_reward=1.5,  # Below minimum
                p_win=0.5,
                expected_value_pct=0.5
            )
    
    def test_position_size_limit(self):
        """Test position size limit enforcement."""
        with pytest.raises(ValueError, match="exceeds 50 bps"):
            TradeRecommendation(
                symbol="AAPL",
                direction=Direction.LONG,
                entry_type="market",
                entry_price=150.0,
                position_size_bps=75,  # Exceeds max
                stop_loss=145.0,
                time_stop_hours=24,
                targets=[155.0],
                thesis="Test",
                risk_reward=2.0,
                p_win=0.5,
                expected_value_pct=1.0
            )


class TestATRSizing:
    """Test ATR-based position sizing."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager(
            max_daily_loss=2000,
            max_position_risk_pct=0.01,
            max_symbol_exposure_pct=0.2,
            max_leverage=2.0,
            per_trade_risk_bps=50,
            max_weekly_loss_pct=0.05
        )
    
    def test_atr_position_sizing(self):
        """Test ATR-based position sizing calculation."""
        # Test with explicit stop
        shares = self.risk_manager.position_size_atr(
            equity=100000,
            entry_price=100,
            stop_price=98,  # 2% stop
            risk_bps=50
        )
        
        # Risk = $50 (0.5% of 100k), Stop distance = $2
        # Shares = 50 / 2 = 25
        assert shares == 25
    
    def test_atr_sizing_with_atr(self):
        """Test sizing with ATR instead of explicit stop."""
        shares = self.risk_manager.position_size_atr(
            equity=100000,
            entry_price=100,
            stop_price=0,  # No explicit stop
            atr=2.5,  # Use ATR
            risk_bps=50,
            is_trend_following=True
        )
        
        # Stop distance = 1.2 * 2.5 = 3.0
        # Risk = $50, Shares = 50 / 3 = 16
        assert shares == 16
    
    def test_position_size_caps(self):
        """Test that position sizing respects exposure limits."""
        shares = self.risk_manager.position_size_atr(
            equity=100000,
            entry_price=1000,  # High price stock
            stop_price=990,
            risk_bps=50
        )
        
        # Max exposure = 20% of 100k = 20k
        # Max shares by exposure = 20000 / 1000 = 20
        # Risk-based shares = 50 / 10 = 5
        assert shares == 5  # Should use risk-based (smaller)
    
    def test_stop_loss_validation(self):
        """Test stop loss validation."""
        is_valid, msg = self.risk_manager.validate_stop_loss(
            entry_price=100,
            stop_price=98
        )
        assert is_valid is True
        
        # Test too wide stop
        is_valid, msg = self.risk_manager.validate_stop_loss(
            entry_price=100,
            stop_price=90  # 10% stop (too wide)
        )
        assert is_valid is False
        assert "too wide" in msg.lower()
    
    def test_weekly_drawdown_check(self):
        """Test weekly drawdown limit."""
        is_valid, msg = self.risk_manager.validate_order(
            symbol="AAPL",
            order_qty=100,
            price=150,
            equity=100000,
            daily_pnl=-1000,
            weekly_pnl=-6000,  # -6% weekly loss
            current_positions={},
            stop_price=145
        )
        assert is_valid is False
        assert "Weekly loss limit" in msg


class TestLiquidityChecks:
    """Test market metadata and liquidity validation."""
    
    @pytest.mark.asyncio
    async def test_liquidity_validation(self):
        """Test liquidity and spread validation."""
        metadata = MarketMetadata(
            symbol="AAPL",
            adv=50_000_000,  # $50M ADV
            spread_pct=0.001,  # 0.1% spread
            spread_dollars=0.15,
            bid=149.85,
            ask=150.00,
            last_price=150.00,
            volume=1_000_000,
            avg_volume_20d=800_000,
            atr=2.5,
            shortable=True,
            borrow_rate=0.0,
            last_updated=datetime.now()
        )
        
        assert metadata.is_liquid() is True
        assert metadata.get_liquidity_score() > 80
    
    @pytest.mark.asyncio
    async def test_illiquid_rejection(self):
        """Test rejection of illiquid symbols."""
        metadata = MarketMetadata(
            symbol="PENNY",
            adv=500_000,  # $500k ADV (below minimum)
            spread_pct=0.03,  # 3% spread (too wide)
            spread_dollars=0.03,
            bid=0.97,
            ask=1.00,
            last_price=1.00,
            volume=100_000,
            avg_volume_20d=50_000,
            atr=0.05,
            shortable=False,
            borrow_rate=50.0,
            last_updated=datetime.now()
        )
        
        assert metadata.is_liquid() is False
        assert metadata.get_liquidity_score() < 20
    
    def test_execution_cost_estimate(self):
        """Test execution cost estimation."""
        ib_client = Mock()
        provider = MarketMetaProvider(ib_client)
        
        # Add mock metadata
        provider._cache["AAPL"] = MarketMetadata(
            symbol="AAPL",
            adv=50_000_000,
            spread_pct=0.001,
            spread_dollars=0.15,
            bid=149.85,
            ask=150.00,
            last_price=150.00,
            volume=1_000_000,
            avg_volume_20d=800_000,
            atr=2.5,
            shortable=True,
            borrow_rate=0.0,
            last_updated=datetime.now()
        )
        
        # Small order - mainly spread cost
        cost_bps = provider.get_execution_cost_estimate("AAPL", 100, is_aggressive=False)
        assert 0 < cost_bps < 10
        
        # Large order - includes impact
        cost_bps = provider.get_execution_cost_estimate("AAPL", 100000, is_aggressive=True)
        assert cost_bps > 10  # Should include market impact


class TestCorrelationBudget:
    """Test correlation and sector exposure control."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.correlation = CorrelationBudget(max_bucket_exposure_pct=0.35)
    
    def test_sector_mapping(self):
        """Test default sector mappings."""
        assert self.correlation.get_symbol_bucket("AAPL") == Sector.TECHNOLOGY.value
        assert self.correlation.get_symbol_bucket("JPM") == Sector.FINANCIALS.value
        assert self.correlation.get_symbol_bucket("UNKNOWN") == "single_UNKNOWN"
    
    def test_exposure_tracking(self):
        """Test exposure tracking by bucket."""
        # Add tech positions
        self.correlation.update_position("AAPL", 100, 150, 100000)
        self.correlation.update_position("GOOGL", 50, 2000, 100000)
        
        # Check tech bucket
        tech_bucket = self.correlation.buckets.get(Sector.TECHNOLOGY.value)
        assert tech_bucket is not None
        assert tech_bucket.total_notional == 15000 + 100000
        assert tech_bucket.percent_of_portfolio > 1.0  # Over 100% (leverage)
    
    def test_correlation_limit_enforcement(self):
        """Test that correlation limits are enforced."""
        # Fill tech bucket to near limit
        self.correlation.update_position("AAPL", 100, 150, 100000)
        self.correlation.update_position("GOOGL", 10, 2000, 100000)
        
        # Try to add more tech
        is_allowed, reason, current_exposure = self.correlation.check_new_position(
            "MSFT", 100, 200, 100000
        )
        
        assert is_allowed is False
        assert "exceed" in reason
        assert current_exposure >= 0.35
    
    def test_available_buckets(self):
        """Test identification of available buckets."""
        # Fill tech bucket
        self.correlation.buckets[Sector.TECHNOLOGY.value] = Mock(percent_of_portfolio=0.30)
        
        available = self.correlation.get_available_buckets()
        
        assert Sector.TECHNOLOGY.value in available  # Still has some room
        assert Sector.FINANCIALS.value in available
        assert Sector.HEALTHCARE.value in available


class TestEVCalculation:
    """Test Expected Value and edge calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.edge_calc = EdgeCalculator()
    
    def test_ev_calculation(self):
        """Test basic EV calculation."""
        ev = self.edge_calc.calculate_ev(
            symbol="AAPL",
            setup_type="breakout",
            conviction=0.75,
            current_volatility=20,
            spread_bps=5,
            custom_p_win=0.55,
            custom_rr=2.0
        )
        
        assert ev.net_ev_pct > 0  # Should be positive EV
        assert ev.risk_reward_ratio == 2.0
        assert 0.4 < ev.p_win < 0.85  # Adjusted by conviction
        assert ev.meets_threshold(min_ev_pct=0, min_rr=1.8)
    
    def test_negative_ev_rejection(self):
        """Test rejection of negative EV trades."""
        ev = self.edge_calc.calculate_ev(
            symbol="AAPL",
            setup_type="momentum",
            conviction=0.3,  # Low conviction
            current_volatility=40,  # High vol
            spread_bps=50,  # Wide spread
            custom_p_win=0.40,
            custom_rr=1.5
        )
        
        assert ev.net_ev_pct < 0  # Should be negative
        assert not ev.meets_threshold(min_ev_pct=0, min_rr=1.8)
    
    def test_kelly_sizing(self):
        """Test Kelly criterion position sizing."""
        ev = EVCalculation(
            symbol="AAPL",
            p_win=0.60,
            avg_win_pct=3.0,
            avg_loss_pct=1.5,
            commission_bps=0.5,
            slippage_bps=2.0,
            time_decay_factor=1.0
        ).calculate()
        
        assert ev.kelly_fraction > 0
        assert ev.kelly_fraction <= 0.25  # Capped at 25% Kelly
        
        # Calculate optimal size
        optimal_bps = self.edge_calc.calculate_optimal_size(ev, max_risk_bps=50)
        assert 0 < optimal_bps <= 50
    
    def test_setup_performance_tracking(self):
        """Test tracking of setup performance."""
        # Record some trades
        self.edge_calc.update_performance("breakout", 2.5, True)
        self.edge_calc.update_performance("breakout", -1.2, False)
        self.edge_calc.update_performance("momentum", 1.8, True)
        
        # Get stats
        breakout_stats = self.edge_calc.get_setup_stats("breakout")
        assert breakout_stats["total_trades"] == 2
        assert breakout_stats["win_rate"] == 0.5
        assert breakout_stats["total_pnl"] == 1.3


class TestCalibration:
    """Test calibration and Brier score tracking."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.calibration = CalibrationTracker(database=Mock())
    
    def test_brier_score_calculation(self):
        """Test Brier score calculation."""
        # Perfect calibration
        predictions = [0.0, 0.5, 1.0]
        outcomes = [False, True, True]  # 50% outcome matches 50% prediction
        brier = self.calibration.calculate_brier_score(predictions, outcomes)
        assert brier < 0.3  # Good calibration
        
        # Poor calibration
        predictions = [0.9, 0.9, 0.9]
        outcomes = [False, False, False]
        brier = self.calibration.calculate_brier_score(predictions, outcomes)
        assert brier > 0.7  # Poor calibration
    
    def test_reliability_calculation(self):
        """Test reliability (calibration plot) calculation."""
        # Well-calibrated predictions
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, 100)
        outcomes = np.random.random(100) < predictions  # Calibrated outcomes
        
        reliability, bin_data = self.calibration.calculate_reliability(
            predictions.tolist(),
            outcomes.tolist()
        )
        
        assert 0.5 < reliability < 1.5  # Should be near 1.0
        assert len(bin_data['centers']) > 0
    
    def test_resolution_calculation(self):
        """Test resolution (discrimination) calculation."""
        # High resolution - clear separation
        predictions = [0.1] * 50 + [0.9] * 50
        outcomes = [False] * 50 + [True] * 50
        
        resolution = self.calibration.calculate_resolution(predictions, outcomes)
        assert resolution > 0.1  # Good discrimination
        
        # Low resolution - no discrimination
        predictions = [0.5] * 100
        outcomes = [False] * 50 + [True] * 50
        
        resolution = self.calibration.calculate_resolution(predictions, outcomes)
        assert resolution < 0.01  # Poor discrimination
    
    def test_calibration_tracking(self):
        """Test tracking of predictions for calibration."""
        # Track some predictions
        self.calibration.track_prediction(0.7, True)
        self.calibration.track_prediction(0.3, False)
        self.calibration.track_prediction(0.8, True)
        self.calibration.track_prediction(0.2, False)
        
        assert len(self.calibration.recent_predictions) == 4
        
        # Check that old predictions are removed
        for _ in range(1000):
            self.calibration.track_prediction(0.5, True)
        
        assert len(self.calibration.recent_predictions) == 1000


class TestLLMClient:
    """Test the decisive LLM client."""
    
    @pytest.mark.asyncio
    async def test_llm_decision_generation(self):
        """Test that LLM client generates valid decisions."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            client = DecisiveLLMClient()
            
            # Mock the Anthropic response
            mock_response = Mock()
            mock_tool_use = Mock()
            mock_tool_use.type = 'tool_use'
            mock_tool_use.input = {
                'mode': 'trade',
                'timestamp_utc': datetime.now().isoformat(),
                'universe_checked': ['AAPL'],
                'conviction': 75,
                'recommendation': {
                    'symbol': 'AAPL',
                    'direction': 'long',
                    'entry_type': 'limit',
                    'entry_price': 150.0,
                    'position_size_bps': 30,
                    'stop_loss': 145.0,
                    'time_stop_hours': 48,
                    'targets': [155.0, 160.0],
                    'thesis': 'Test trade',
                    'risk_reward': 2.0,
                    'p_win': 0.55,
                    'expected_value_pct': 1.5
                },
                'compliance_checks': {
                    'liquidity_ok': True,
                    'spread_ok': True,
                    'borrow_ok': True,
                    'correlation_ok': True
                },
                'risk_state': {
                    'day_dd_bps': 0,
                    'week_dd_bps': 0,
                    'cash_pct': 100.0,
                    'open_positions': 0,
                    'total_exposure_pct': 0
                },
                'costs': {
                    'commission_bps': 0.5,
                    'slippage_bps': 2.0,
                    'total_bps': 2.5
                },
                'watchlist': [],
                'notes': 'Test decision'
            }
            mock_response.content = [mock_tool_use]
            
            client.client.beta.tools.messages.create = AsyncMock(return_value=mock_response)
            
            # Generate decision
            market_data = {
                'AAPL': MarketData(
                    symbol='AAPL',
                    price=150.0,
                    volume=1000000,
                    avg_volume=800000,
                    atr=2.5,
                    adv=120000000,
                    spread_pct=0.001,
                    shortable=True,
                    borrow_rate=0.0
                )
            }
            
            decision = await client.get_trading_decision(
                market_data=market_data,
                news_events=[],
                aggressiveness_level=1
            )
            
            assert decision.mode == TradingMode.TRADE
            assert decision.conviction == 75
            assert decision.should_execute() is True
            assert decision.recommendation.symbol == 'AAPL'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])