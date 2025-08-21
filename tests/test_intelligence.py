"""
Unit tests for intelligence module
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from robo_trader.intelligence import ClaudeTrader, KellyCriterion


@pytest.mark.asyncio
async def test_claude_trader_initialization():
    """Test ClaudeTrader initialization"""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
        trader = ClaudeTrader()
        assert trader.model == "claude-3-5-sonnet-latest"


@pytest.mark.asyncio
async def test_claude_trader_no_api_key():
    """Test ClaudeTrader fails without API key"""
    with patch.dict('os.environ', {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
            ClaudeTrader()


@pytest.mark.asyncio
async def test_analyze_market_event():
    """Test market event analysis"""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
        with patch('robo_trader.intelligence.AsyncAnthropic') as mock_anthropic:
            # Mock Claude response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"direction": "bullish", "conviction": 75, "rationale": "test", "entry_price": 100, "stop_loss": 95, "take_profit": 110}')]
            
            mock_client = AsyncMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client
            
            trader = ClaudeTrader()
            signal = await trader.analyze_market_event(
                "Fed raises rates by 25bps",
                "SPY",
                {"price": 450.0, "volume": 1000000}
            )
            
            assert signal['direction'] == 'bullish'
            assert signal['conviction'] == 75
            assert signal['entry_price'] == 100
            assert 'rationale' in signal


@pytest.mark.asyncio
async def test_analyze_market_event_error_handling():
    """Test error handling in market analysis"""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
        with patch('robo_trader.intelligence.AsyncAnthropic') as mock_anthropic:
            # Mock API error
            mock_client = AsyncMock()
            mock_client.messages.create.side_effect = Exception("API Error")
            mock_anthropic.return_value = mock_client
            
            trader = ClaudeTrader()
            signal = await trader.analyze_market_event(
                "Test event",
                "TEST",
                {"price": 100}
            )
            
            assert signal['direction'] == 'neutral'
            assert signal['conviction'] == 0
            assert signal['error'] == True


def test_kelly_criterion_position_sizing():
    """Test Kelly Criterion calculations"""
    # Test with positive edge
    size = KellyCriterion.calculate_position_size(
        win_probability=0.6,
        avg_win_return=0.05,
        avg_loss_return=-0.02,
        kelly_fraction=0.25
    )
    assert 0 < size <= 0.1  # Should be positive but capped at 10%
    
    # Test with no edge
    size = KellyCriterion.calculate_position_size(
        win_probability=0.4,
        avg_win_return=0.05,
        avg_loss_return=-0.05,
        kelly_fraction=0.25
    )
    assert size == 0  # Should be 0 with negative expectancy
    
    # Test with invalid loss (no risk)
    size = KellyCriterion.calculate_position_size(
        win_probability=0.6,
        avg_win_return=0.05,
        avg_loss_return=0.01,  # Positive "loss"
        kelly_fraction=0.25
    )
    assert size == 0


def test_conviction_based_sizing():
    """Test conviction-based position sizing"""
    # Low conviction - no trade
    assert KellyCriterion.size_from_conviction(30) == 0
    assert KellyCriterion.size_from_conviction(49) == 0
    
    # Minimum conviction - base size
    assert KellyCriterion.size_from_conviction(50) == 0.02
    
    # Mid conviction
    size_70 = KellyCriterion.size_from_conviction(70)
    assert 0.02 < size_70 < 0.10
    
    # Max conviction - max size
    assert KellyCriterion.size_from_conviction(100) == 0.10
    
    # Test scaling is monotonic
    size_60 = KellyCriterion.size_from_conviction(60)
    size_80 = KellyCriterion.size_from_conviction(80)
    assert size_60 < size_70 < size_80


def test_json_extraction():
    """Test JSON extraction from Claude responses"""
    trader = ClaudeTrader.__new__(ClaudeTrader)
    
    # Test with code blocks
    content = '```json\n{"direction": "bullish", "conviction": 80}\n```'
    result = trader._extract_json_from_response(content)
    assert result['direction'] == 'bullish'
    assert result['conviction'] == 80
    
    # Test without code blocks
    content = '{"direction": "bearish", "conviction": 60}'
    result = trader._extract_json_from_response(content)
    assert result['direction'] == 'bearish'
    assert result['conviction'] == 60
    
    # Test with surrounding text
    content = 'Here is my analysis: {"direction": "neutral", "conviction": 30} based on...'
    result = trader._extract_json_from_response(content)
    assert result['direction'] == 'neutral'
    assert result['conviction'] == 30
    
    # Test with invalid JSON
    content = 'This is not JSON'
    result = trader._extract_json_from_response(content)
    assert result['direction'] == 'neutral'
    assert result['conviction'] == 0
    assert result['parse_error'] == True


def test_signal_validation():
    """Test signal validation"""
    trader = ClaudeTrader.__new__(ClaudeTrader)
    
    # Valid signal
    valid_signal = {'direction': 'bullish', 'conviction': 75}
    assert trader._validate_signal(valid_signal) == True
    
    # Missing direction
    invalid_signal = {'conviction': 75}
    assert trader._validate_signal(invalid_signal) == False
    
    # Missing conviction
    invalid_signal = {'direction': 'bullish'}
    assert trader._validate_signal(invalid_signal) == False
    
    # Empty signal
    assert trader._validate_signal({}) == False