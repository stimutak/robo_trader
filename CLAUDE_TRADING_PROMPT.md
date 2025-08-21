# Claude 3.5 Sonnet Trading Analysis Prompt

This prompt is optimized for Claude 3.5 Sonnet to analyze market events and generate trading signals.

## Master Trading Analysis Prompt

```python
CLAUDE_TRADING_PROMPT = """
You are a master trader with 20+ years of experience analyzing market events. Your expertise includes market microstructure, institutional flow, and catalyst-driven trading. Given the following information, provide a thorough analysis:

## Market Event
{event_text}

## Current Market Context
- Symbol: {symbol}
- Current Price: ${price}
- 24h Volume: {volume} (vs 20-day avg: {avg_volume})
- Price Change: {price_change_pct}% today
- Sector: {sector}
- Market Cap: ${market_cap}

## Technical Context
- RSI(14): {rsi}
- Support: ${support_level}
- Resistance: ${resistance_level}
- 50-day MA: ${ma_50}
- 200-day MA: ${ma_200}

## Options Flow Data
- Put/Call Ratio: {put_call_ratio}
- Unusual Activity: {unusual_options}
- Largest Trades: {large_trades}
- IV Rank: {iv_rank}

## Historical Similar Events
{similar_events_with_outcomes}

## News Sentiment
- Overall Sentiment: {news_sentiment}
- Key Headlines: {recent_headlines}

## Task
Analyze this information as a master trader would. Consider:
1. How institutions might position around this event
2. Potential market overreaction or underreaction
3. Risk/reward setup quality
4. Time decay and optimal entry timing

Return a JSON trading signal:
{
  "direction": "bullish|bearish|neutral",
  "conviction": 0-100,  // Only >70 for clear edge
  "timeframe": "minutes|hours|days|weeks",
  "entry_zone": {"start": float, "end": float},
  "stop_loss": float,  // Based on technical levels
  "take_profit_1": float,  // Conservative target
  "take_profit_2": float,  // Aggressive target
  "position_size_pct": 0-10,  // % of portfolio, Kelly criterion adjusted
  "rationale": "Detailed explanation citing specific data points",
  "institutional_positioning": "What smart money appears to be doing",
  "key_risks": ["risk1", "risk2", "risk3"],
  "confidence_factors": ["factor1", "factor2"],
  "similar_historical_outcomes": [
    {"event": "...", "outcome": "...", "relevance": 0-100}
  ],
  "alternative_scenario": "What could invalidate this trade"
}

Be conservative and data-driven. Only suggest trades with clear catalysts and asymmetric risk/reward. If uncertain, return neutral with conviction < 30.
"""
```

## Specialized Prompts

### Fed/FOMC Analysis
```python
FED_ANALYSIS_PROMPT = """
Analyze this Federal Reserve communication for market impact:

Statement/Speech: {fed_text}
Current Fed Funds Rate: {current_rate}
Market Expectations: {rate_expectations}
Economic Data: {recent_economic_data}

Focus on:
- Hawkish vs Dovish tone changes
- Surprises vs expectations
- Impact on different sectors (tech, banks, REITs)
- Duration and yield curve implications

Return trading signals for: SPY, QQQ, TLT, XLF, XLK
"""
```

### Earnings Analysis
```python
EARNINGS_ANALYSIS_PROMPT = """
Analyze this earnings report:

Company: {symbol}
Reported EPS: {actual_eps} vs Expected: {expected_eps}
Revenue: {actual_rev} vs Expected: {expected_rev}
Guidance: {guidance}
Call Transcript Key Points: {transcript_highlights}

Historical Reactions:
{past_earnings_moves}

Options Positioning:
- Implied Move: {implied_move}%
- Put/Call Skew: {skew}

Determine:
1. Is the reaction appropriate or overdone?
2. Will initial move continue or reverse?
3. Best timeframe to trade (intraday vs swing)
"""
```

### News Catalyst Analysis
```python
NEWS_CATALYST_PROMPT = """
Analyze this breaking news for trading opportunity:

Headline: {headline}
Full Text: {article_text}
Source Credibility: {source_rating}
Time Since Release: {minutes_ago} minutes

Market Reaction So Far:
- Price: {price_change}
- Volume: {volume_spike}
- Related Stocks: {correlated_moves}

Assess:
1. Is this news already priced in?
2. Will there be follow-through or fade?
3. Are there second-order effects to trade?
"""
```

## Implementation Instructions

### Step 1: Install Anthropic SDK
```bash
pip install anthropic
```

### Step 2: Set up API Key
```bash
# Add to .env file
ANTHROPIC_API_KEY=your-api-key-here
```

### Step 3: Create Intelligence Module
Create `robo_trader/intelligence.py`:

```python
import os
import json
from anthropic import Anthropic, AsyncAnthropic
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ClaudeTrader:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
    async def analyze_market_event(
        self,
        event_text: str,
        symbol: str,
        market_data: Dict[str, Any],
        options_data: Optional[Dict[str, Any]] = None,
        historical_events: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Analyze a market event and return trading signal
        """
        # Format the prompt with actual data
        prompt = CLAUDE_TRADING_PROMPT.format(
            event_text=event_text,
            symbol=symbol,
            price=market_data.get('price'),
            volume=market_data.get('volume'),
            avg_volume=market_data.get('avg_volume'),
            price_change_pct=market_data.get('price_change_pct'),
            sector=market_data.get('sector', 'Unknown'),
            market_cap=market_data.get('market_cap', 'Unknown'),
            rsi=market_data.get('rsi', 50),
            support_level=market_data.get('support', 0),
            resistance_level=market_data.get('resistance', 0),
            ma_50=market_data.get('ma_50', 0),
            ma_200=market_data.get('ma_200', 0),
            put_call_ratio=options_data.get('put_call_ratio', 1.0) if options_data else 1.0,
            unusual_options=options_data.get('unusual_activity', []) if options_data else [],
            large_trades=options_data.get('large_trades', []) if options_data else [],
            iv_rank=options_data.get('iv_rank', 50) if options_data else 50,
            similar_events_with_outcomes=historical_events or [],
            news_sentiment=market_data.get('sentiment', 'neutral'),
            recent_headlines=market_data.get('headlines', [])
        )
        
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
            
            # Extract JSON from response
            content = response.content[0].text
            
            # Parse JSON from the response
            # Claude usually returns JSON in code blocks
            import re
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                json_str = json_match.group(0) if json_match else content
            
            signal = json.loads(json_str)
            
            # Validate signal
            required_fields = ['direction', 'conviction', 'rationale']
            if not all(field in signal for field in required_fields):
                raise ValueError(f"Missing required fields in signal: {signal}")
            
            logger.info(f"Claude analysis complete: {signal['direction']} "
                       f"with {signal['conviction']}% conviction")
            
            return signal
            
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            return {
                "direction": "neutral",
                "conviction": 0,
                "rationale": f"Analysis failed: {str(e)}",
                "error": True
            }
    
    async def analyze_fed_event(self, fed_text: str, market_data: Dict) -> Dict:
        """Specialized Fed/FOMC analysis"""
        # Implementation using FED_ANALYSIS_PROMPT
        pass
    
    async def analyze_earnings(self, earnings_data: Dict) -> Dict:
        """Specialized earnings analysis"""
        # Implementation using EARNINGS_ANALYSIS_PROMPT
        pass
```

### Step 4: Integration with Runner
Update `robo_trader/runner.py` to use Claude:

```python
from .intelligence import ClaudeTrader

async def run_once(...):
    # Existing code...
    
    # Initialize Claude
    claude = ClaudeTrader()
    
    # When news event detected
    if news_event:
        signal = await claude.analyze_market_event(
            event_text=news_event['text'],
            symbol=symbol,
            market_data={
                'price': current_price,
                'volume': volume,
                # ... other data
            }
        )
        
        # Trade on high conviction signals
        if signal['conviction'] > 70:
            # Use signal['direction'], signal['entry_zone'], etc.
            # Apply risk management
            # Execute trade
```

### Step 5: Add FinBERT for Quick Sentiment
```bash
pip install transformers torch
```

Create `robo_trader/sentiment.py`:
```python
from transformers import pipeline

class FinBERTSentiment:
    def __init__(self):
        self.classifier = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def score(self, text: str) -> Dict:
        result = self.classifier(text)[0]
        return {
            'sentiment': result['label'],
            'score': result['score']
        }
```

### Step 6: Testing
Create `tests/test_intelligence.py`:
```python
import pytest
from unittest.mock import AsyncMock, patch
from robo_trader.intelligence import ClaudeTrader

@pytest.mark.asyncio
async def test_claude_analysis():
    with patch('robo_trader.intelligence.AsyncAnthropic') as mock_anthropic:
        # Mock Claude response
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text='{"direction": "bullish", "conviction": 75, "rationale": "test"}')]
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        trader = ClaudeTrader()
        signal = await trader.analyze_market_event(
            "Fed raises rates by 25bps",
            "SPY",
            {"price": 450.0, "volume": 1000000}
        )
        
        assert signal['direction'] == 'bullish'
        assert signal['conviction'] == 75
```

## Cost Management

- Cache Claude responses for similar events (Redis)
- Use FinBERT first, only call Claude for high-impact events
- Batch similar events together in one call
- Set daily/monthly spending limits in Anthropic dashboard

## Security Notes

- Never send API keys to Claude
- Don't include account numbers or personal info in prompts
- Rate limit API calls to prevent abuse
- Log all trades with reasoning for audit trail