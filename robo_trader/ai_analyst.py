"""
AI Market Analyst using LLMs for intelligent trading decisions.

Integrates with OpenAI/Anthropic APIs to analyze market events, news,
and provide trading insights like a master trader.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class MarketSentiment(Enum):
    """Market sentiment levels from AI analysis."""

    VERY_BULLISH = 2
    BULLISH = 1
    NEUTRAL = 0
    BEARISH = -1
    VERY_BEARISH = -2


@dataclass
class MarketAnalysis:
    """Result of AI market analysis."""

    symbol: str
    sentiment: MarketSentiment
    confidence: float  # 0.0 to 1.0
    reasoning: str
    key_factors: List[str]
    risk_level: str  # "low", "medium", "high"
    suggested_action: str  # "buy", "sell", "hold", "wait"
    timestamp: datetime


class AIAnalyst:
    """
    AI-powered market analyst using LLMs.

    Features:
    - Analyze news and market events
    - Predict market reactions
    - Identify trading opportunities
    - Assess risk levels
    """

    def __init__(self, provider: str = "openai", model: str = None):
        """
        Initialize AI analyst.

        Args:
            provider: "openai" or "anthropic"
            model: Specific model to use (e.g., "gpt-4", "claude-3-opus")
        """
        self.provider = provider
        self.model = model or self._get_default_model()
        self.api_key = self._load_api_key()
        self.client = self._init_client()

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        if self.provider == "openai":
            return "gpt-4-turbo-preview"
        elif self.provider == "anthropic":
            return "claude-3-haiku-20240307"  # Claude 3 Haiku (fastest, cheapest)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment."""
        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
        elif self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
        else:
            key = None

        if not key:
            logger.warning(f"No API key found for {self.provider}. AI analysis disabled.")

        return key

    def _init_client(self):
        """Initialize API client."""
        if not self.api_key:
            return None

        try:
            if self.provider == "openai":
                import openai

                openai.api_key = self.api_key
                return openai
            elif self.provider == "anthropic":
                from anthropic import Anthropic

                return Anthropic(api_key=self.api_key)
        except ImportError as e:
            logger.warning(f"Could not import {self.provider} library: {e}")
            return None

    def analyze_market_event(
        self, symbol: str, event_text: str, market_data: Optional[Dict] = None
    ) -> MarketAnalysis:
        """
        Analyze a market event and predict impact.

        Args:
            symbol: Stock symbol
            event_text: News or event description
            market_data: Optional recent price/volume data

        Returns:
            MarketAnalysis with AI insights
        """
        if not self.client:
            return self._default_analysis(symbol)

        prompt = self._build_analysis_prompt(symbol, event_text, market_data)

        try:
            response = self._call_llm(prompt)
            return self._parse_analysis(symbol, response)
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return self._default_analysis(symbol)

    def _build_analysis_prompt(
        self, symbol: str, event_text: str, market_data: Optional[Dict]
    ) -> str:
        """Build prompt for LLM analysis."""
        prompt = f"""You are a master trader analyzing market events. Analyze this event for {symbol}:

EVENT: {event_text}

"""

        if market_data:
            prompt += f"""RECENT MARKET DATA:
- Current Price: ${market_data.get('price', 'N/A')}
- 5-Day Change: {market_data.get('change_5d', 'N/A')}%
- Volume: {market_data.get('volume', 'N/A')}
- RSI: {market_data.get('rsi', 'N/A')}

"""

        prompt += """Provide analysis in JSON format:
{
    "sentiment": "very_bullish/bullish/neutral/bearish/very_bearish",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation",
    "key_factors": ["factor1", "factor2"],
    "risk_level": "low/medium/high",
    "suggested_action": "buy/sell/hold/wait"
}

Focus on:
1. How will smart money react?
2. What will retail traders miss?
3. Time horizon for impact
4. Risk/reward assessment"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.provider == "openai":
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master trader with deep market knowledge.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                system="You are a master trader with deep market knowledge.",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return "{}"

    def _parse_analysis(self, symbol: str, response: str) -> MarketAnalysis:
        """Parse LLM response into MarketAnalysis."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}

            # Map sentiment string to enum
            sentiment_map = {
                "very_bullish": MarketSentiment.VERY_BULLISH,
                "bullish": MarketSentiment.BULLISH,
                "neutral": MarketSentiment.NEUTRAL,
                "bearish": MarketSentiment.BEARISH,
                "very_bearish": MarketSentiment.VERY_BEARISH,
            }

            sentiment = sentiment_map.get(data.get("sentiment", "neutral"), MarketSentiment.NEUTRAL)

            return MarketAnalysis(
                symbol=symbol,
                sentiment=sentiment,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", "No reasoning provided"),
                key_factors=data.get("key_factors", []),
                risk_level=data.get("risk_level", "medium"),
                suggested_action=data.get("suggested_action", "hold"),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return self._default_analysis(symbol)

    def _default_analysis(self, symbol: str) -> MarketAnalysis:
        """Return default neutral analysis."""
        return MarketAnalysis(
            symbol=symbol,
            sentiment=MarketSentiment.NEUTRAL,
            confidence=0.0,
            reasoning="AI analysis unavailable",
            key_factors=[],
            risk_level="medium",
            suggested_action="hold",
            timestamp=datetime.now(),
        )

    def analyze_earnings(self, symbol: str, earnings_data: Dict) -> MarketAnalysis:
        """Analyze earnings report impact."""
        event_text = f"""
        Earnings Report:
        - EPS: ${earnings_data.get('eps', 'N/A')} vs ${earnings_data.get('eps_estimate', 'N/A')} expected
        - Revenue: ${earnings_data.get('revenue', 'N/A')}M vs ${earnings_data.get('revenue_estimate', 'N/A')}M expected
        - Guidance: {earnings_data.get('guidance', 'Not provided')}
        """

        return self.analyze_market_event(symbol, event_text, earnings_data.get("market_data"))

    def analyze_fed_event(self, event_description: str) -> Dict[str, MarketAnalysis]:
        """Analyze Fed announcement impact on multiple sectors."""
        sectors = {
            "SPY": "S&P 500",
            "QQQ": "Tech sector",
            "XLF": "Financials",
            "TLT": "Bonds",
        }

        analyses = {}
        for symbol, sector in sectors.items():
            event_text = f"Fed Event: {event_description}\nAnalyzing impact on {sector}"
            analyses[symbol] = self.analyze_market_event(symbol, event_text)

        return analyses

    def find_opportunities(
        self, news_headlines: List[str], exclude_symbols: List[str] = None
    ) -> List[Dict]:
        """
        Scan news headlines to find buying opportunities.

        Args:
            news_headlines: List of news headlines to analyze
            exclude_symbols: Symbols to exclude (already owned)

        Returns:
            List of opportunities: [{"symbol": "XXX", "action": "buy", "confidence": 0.8, "reason": "..."}]
        """
        if not self.client:
            return []

        exclude_symbols = exclude_symbols or []
        news_text = "\n".join([f"- {h}" for h in news_headlines[:15]])

        prompt = f"""You are a master stock trader scanning today's news for buying opportunities.

NEWS HEADLINES:
{news_text}

ALREADY OWNED (DO NOT RECOMMEND): {', '.join(exclude_symbols) if exclude_symbols else 'None'}

Find stocks mentioned in the news that are STRONG BUY opportunities based on:
1. Positive catalysts (earnings beats, partnerships, product launches, upgrades)
2. Sector momentum (AI boom, clean energy, etc.)
3. Undervalued situations (oversold, beaten down unfairly)

For each opportunity, provide:
- Stock symbol (UPPERCASE, US stocks only)
- Confidence (0.0-1.0, only include if > 0.6)
- Brief reason (1 sentence)

Respond in JSON format:
{{"opportunities": [
  {{"symbol": "XXXX", "confidence": 0.8, "reason": "Strong catalyst..."}},
  ...
]}}

Only include HIGH CONVICTION plays. If no clear opportunities, return empty list.
Be specific - identify actual stock symbols from the news."""

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",  # Fast model for scanning
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            else:  # OpenAI
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content

            # Parse JSON response
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                opportunities = result.get("opportunities", [])
                # Filter out excluded symbols and add action
                valid_opps = []
                for opp in opportunities:
                    symbol = opp.get("symbol", "").upper()
                    if symbol and symbol not in exclude_symbols and opp.get("confidence", 0) > 0.5:
                        valid_opps.append(
                            {
                                "symbol": symbol,
                                "action": "buy",
                                "confidence": opp.get("confidence", 0.7),
                                "reason": opp.get("reason", "AI identified opportunity"),
                            }
                        )
                logger.info(f"AI found {len(valid_opps)} buying opportunities")
                return valid_opps
            return []
        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")
            return []

    def get_trade_conviction(self, analyses: List[MarketAnalysis]) -> Tuple[str, float]:
        """
        Determine overall trade conviction from multiple analyses.

        Returns:
            (action, confidence) tuple
        """
        if not analyses:
            return ("hold", 0.0)

        # Weight by confidence
        weighted_sentiment = 0.0
        total_confidence = 0.0

        for analysis in analyses:
            weighted_sentiment += analysis.sentiment.value * analysis.confidence
            total_confidence += analysis.confidence

        if total_confidence == 0:
            return ("hold", 0.0)

        avg_sentiment = weighted_sentiment / total_confidence
        avg_confidence = total_confidence / len(analyses)

        # Determine action
        if avg_sentiment > 1.0:
            action = "buy"
        elif avg_sentiment < -1.0:
            action = "sell"
        else:
            action = "hold"

        return (action, avg_confidence)


def create_analyst(provider: Optional[str] = None) -> Optional[AIAnalyst]:
    """
    Create AI analyst if API keys are available.

    Args:
        provider: Force specific provider, otherwise auto-detect

    Returns:
        AIAnalyst instance or None if no API keys
    """
    if provider:
        return AIAnalyst(provider)

    # Auto-detect based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Creating OpenAI analyst")
        return AIAnalyst("openai")
    elif os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Creating Anthropic analyst")
        return AIAnalyst("anthropic")
    else:
        logger.warning("No AI API keys found. AI analysis disabled.")
        return None
