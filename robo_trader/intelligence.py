"""
Intelligence module for AI-powered market analysis using Claude 3.5 Sonnet
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from anthropic import AsyncAnthropic
import logging

from .logger import get_logger

logger = get_logger(__name__)

# Master trading prompt for Claude
CLAUDE_TRADING_PROMPT = """
You are a master trader with 20+ years of experience analyzing market events. Your expertise includes market microstructure, institutional flow, and catalyst-driven trading. Given the following information, provide a thorough analysis:

## Market Event
{event_text}

## Current Market Context
- Symbol: {symbol}
- Current Price: ${price}
- 24h Volume: {volume} (vs 20-day avg: {avg_volume})
- Price Change: {price_change_pct}% today

## Technical Context
- RSI(14): {rsi}
- Support: ${support_level}
- Resistance: ${resistance_level}

## Task
Analyze this information as a master trader would. Consider:
1. How institutions might position around this event
2. Potential market overreaction or underreaction
3. Risk/reward setup quality
4. Time decay and optimal entry timing

Return a JSON trading signal:
{{
  "direction": "bullish|bearish|neutral",
  "conviction": 0-100,
  "timeframe": "minutes|hours|days|weeks",
  "entry_price": float,
  "stop_loss": float,
  "take_profit": float,
  "position_size_pct": 0-10,
  "rationale": "Detailed explanation",
  "key_risks": ["risk1", "risk2"],
  "alternative_scenario": "What could invalidate this trade"
}}

Be conservative and data-driven. Only suggest trades with clear catalysts and asymmetric risk/reward. If uncertain, return neutral with conviction < 30.
"""

FED_ANALYSIS_PROMPT = """
Analyze this Federal Reserve communication for market impact:

Statement/Speech: {fed_text}
Current Fed Funds Rate: {current_rate}
Market Expectations: {rate_expectations}

Focus on:
- Hawkish vs Dovish tone changes
- Surprises vs expectations
- Impact on different sectors (tech, banks, REITs)

Return trading signals for: SPY, QQQ, TLT, XLF
"""

EARNINGS_ANALYSIS_PROMPT = """
Analyze this earnings report:

Company: {symbol}
Reported EPS: {actual_eps} vs Expected: {expected_eps}
Revenue: {actual_rev} vs Expected: {expected_rev}
Guidance: {guidance}

Determine:
1. Is the reaction appropriate or overdone?
2. Will initial move continue or reverse?
3. Best timeframe to trade (intraday vs swing)
"""


class ClaudeTrader:
    """Claude 3.5 Sonnet integration for market analysis"""
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment. Please add it to .env file")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-latest"  # Latest Claude 3.5 Sonnet model
        logger.info("ClaudeTrader initialized with model: %s", self.model)
        
    async def analyze_market_event(
        self,
        event_text: str,
        symbol: str,
        market_data: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
        historical_events: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Analyze a market event and return trading signal
        
        Args:
            event_text: The news or event to analyze
            symbol: Stock symbol
            market_data: Current market data (price, volume, etc)
            options_data: Optional options flow data
            historical_events: Optional similar historical events
            
        Returns:
            Trading signal with direction, conviction, and rationale
        """
        # Fill in defaults for missing data
        market_data = market_data or {}
        
        # Format the prompt with actual data
        prompt = CLAUDE_TRADING_PROMPT.format(
            event_text=event_text,
            symbol=symbol,
            price=market_data.get('price', 0),
            volume=market_data.get('volume', 0),
            avg_volume=market_data.get('avg_volume', market_data.get('volume', 0)),
            price_change_pct=market_data.get('price_change_pct', 0),
            rsi=market_data.get('rsi', 50),
            support_level=market_data.get('support', 0),
            resistance_level=market_data.get('resistance', 0)
        )
        
        logger.info(f"Analyzing market event for {symbol}: {event_text[:100]}...")
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for consistency
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract text from response
            content = response.content[0].text
            
            # Parse JSON from the response
            signal = self._extract_json_from_response(content)
            
            # Validate signal
            if not self._validate_signal(signal):
                raise ValueError(f"Invalid signal format: {signal}")
            
            logger.info(f"Claude analysis complete for {symbol}: {signal['direction']} "
                       f"with {signal['conviction']}% conviction")
            
            return signal
            
        except Exception as e:
            logger.error(f"Claude analysis failed for {symbol}: {e}")
            return {
                "direction": "neutral",
                "conviction": 0,
                "rationale": f"Analysis failed: {str(e)}",
                "error": True
            }
    
    async def analyze_fed_event(
        self, 
        fed_text: str, 
        current_rate: float,
        rate_expectations: str
    ) -> Dict[str, Any]:
        """
        Specialized Fed/FOMC analysis
        
        Args:
            fed_text: Fed statement or speech text
            current_rate: Current Fed funds rate
            rate_expectations: Market expectations
            
        Returns:
            Trading signals for major indices
        """
        prompt = FED_ANALYSIS_PROMPT.format(
            fed_text=fed_text,
            current_rate=current_rate,
            rate_expectations=rate_expectations
        )
        
        logger.info("Analyzing Fed event...")
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            signal = self._extract_json_from_response(content)
            
            logger.info("Fed analysis complete")
            return signal
            
        except Exception as e:
            logger.error(f"Fed analysis failed: {e}")
            return {"error": True, "message": str(e)}
    
    async def analyze_earnings(
        self,
        symbol: str,
        actual_eps: float,
        expected_eps: float,
        actual_rev: float,
        expected_rev: float,
        guidance: str
    ) -> Dict[str, Any]:
        """
        Specialized earnings analysis
        
        Args:
            symbol: Stock symbol
            actual_eps: Reported EPS
            expected_eps: Expected EPS
            actual_rev: Reported revenue
            expected_rev: Expected revenue
            guidance: Forward guidance text
            
        Returns:
            Trading signal for earnings reaction
        """
        prompt = EARNINGS_ANALYSIS_PROMPT.format(
            symbol=symbol,
            actual_eps=actual_eps,
            expected_eps=expected_eps,
            actual_rev=actual_rev,
            expected_rev=expected_rev,
            guidance=guidance
        )
        
        logger.info(f"Analyzing earnings for {symbol}...")
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            signal = self._extract_json_from_response(content)
            
            logger.info(f"Earnings analysis complete for {symbol}")
            return signal
            
        except Exception as e:
            logger.error(f"Earnings analysis failed for {symbol}: {e}")
            return {"error": True, "message": str(e)}
    
    def _extract_json_from_response(self, content: str) -> Dict[str, Any]:
        """Extract JSON from Claude's response"""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # If no JSON found, try to parse the whole content
                json_str = content
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Claude response: {e}")
            # Return a basic structure from the text
            return {
                "direction": "neutral",
                "conviction": 0,
                "rationale": content,
                "parse_error": True
            }
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate that signal has required fields"""
        required_fields = ['direction', 'conviction']
        return all(field in signal for field in required_fields)


class KellyCriterion:
    """Kelly Criterion for optimal position sizing based on edge"""
    
    @staticmethod
    def calculate_position_size(
        win_probability: float,
        avg_win_return: float,
        avg_loss_return: float,
        kelly_fraction: float = 0.25  # Use 25% of Kelly for safety
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Args:
            win_probability: Probability of winning (0-1)
            avg_win_return: Average return when winning (e.g., 0.05 for 5%)
            avg_loss_return: Average loss when losing (e.g., -0.02 for -2%)
            kelly_fraction: Fraction of Kelly to use (default 0.25 for safety)
            
        Returns:
            Optimal position size as fraction of capital (0-1)
        """
        if avg_loss_return >= 0:
            return 0  # No edge if we don't have a proper loss scenario
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        q = 1 - win_probability
        b = abs(avg_win_return / avg_loss_return)
        
        kelly_full = (win_probability * b - q) / b
        
        # Apply safety fraction and constraints
        position_size = kelly_full * kelly_fraction
        
        # Constrain between 0 and 0.1 (max 10% per position)
        return max(0, min(0.1, position_size))
    
    @staticmethod
    def size_from_conviction(
        conviction: int,  # 0-100
        base_size: float = 0.02,  # 2% base position
        max_size: float = 0.10  # 10% max position
    ) -> float:
        """
        Simple conviction-based sizing
        
        Args:
            conviction: Conviction score 0-100
            base_size: Minimum position size
            max_size: Maximum position size
            
        Returns:
            Position size as fraction of capital
        """
        if conviction < 50:
            return 0  # Don't trade low conviction
        
        # Linear scaling from base to max based on conviction
        # 50% conviction = base_size, 100% conviction = max_size
        scale = (conviction - 50) / 50
        position_size = base_size + (max_size - base_size) * scale
        
        return round(position_size, 3)