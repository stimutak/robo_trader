"""
Sentiment analysis module for financial news
Using lightweight approach initially, can upgrade to FinBERT later
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    sentiment: str  # bullish, bearish, neutral
    score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    keywords: List[str]  # Key words that influenced the score


class SimpleSentimentAnalyzer:
    """
    Lightweight rule-based sentiment analyzer for financial text.
    Fast pre-filter before sending to Claude for deep analysis.
    """
    
    def __init__(self):
        # Financial sentiment keywords with weights
        self.bullish_keywords = {
            # Strong bullish
            'beat': 3.0, 'beats': 3.0, 'exceeded': 3.0, 'surpass': 3.0,
            'surge': 3.0, 'surges': 3.0, 'soar': 3.0, 'rally': 3.0, 'breakout': 3.0,
            'upgrade': 2.5, 'raised': 2.0, 'strong': 2.0, 'buy': 2.5,
            'outperform': 2.5, 'bullish': 3.0, 'boom': 2.5,
            
            # Moderate bullish
            'growth': 1.5, 'expand': 1.5, 'improve': 1.5, 'gain': 1.5,
            'rise': 1.2, 'increase': 1.2, 'up': 1.0, 'positive': 1.5,
            'recovery': 2.0, 'rebound': 2.0, 'optimistic': 2.0,
            
            # Fed/Policy bullish
            'dovish': 2.5, 'stimulus': 2.5, 'easing': 2.5, 'accommodative': 2.0,
            'cut rates': 3.0, 'lower rates': 2.5, 'pause': 1.5,
        }
        
        self.bearish_keywords = {
            # Strong bearish
            'miss': -3.0, 'misses': -3.0, 'missed': -3.0, 'disappoint': -3.0, 
            'disappoints': -3.0, 'disappointing': -3.0,
            'plunge': -3.0, 'plunges': -3.0, 'crash': -3.0, 'collapse': -3.0, 
            'slump': -2.5, 'tumble': -2.5,
            'downgrade': -2.5, 'cut': -2.0, 'cuts': -2.0, 'slash': -2.5, 
            'weak': -2.0, 'sell': -2.5, 'underperform': -2.5, 'bearish': -3.0,
            
            # Moderate bearish  
            'decline': -1.5, 'fall': -1.5, 'drop': -1.5, 'decrease': -1.2,
            'down': -1.0, 'negative': -1.5, 'concern': -1.2, 'worry': -1.2,
            'risk': -1.0, 'threat': -1.5, 'warning': -2.0,
            
            # Fed/Policy bearish
            'hawkish': -2.5, 'tighten': -2.5, 'hike rates': -3.0, 'hikes': -2.5,
            'raise rates': -2.5, 'restrictive': -2.0, 'inflation': -1.5,
        }
        
        # Context modifiers
        self.negation_words = {'not', 'no', 'never', 'neither', 'nor', 'without'}
        self.amplifiers = {'very': 1.5, 'extremely': 2.0, 'slightly': 0.5, 'somewhat': 0.7}
        
    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Financial news or announcement text
            
        Returns:
            SentimentScore with sentiment, score, and confidence
        """
        if not text:
            return SentimentScore('neutral', 0.0, 0.0, [])
        
        # Preprocess text
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]+\b', text_lower)
        
        # Calculate sentiment scores
        bullish_score = 0.0
        bearish_score = 0.0
        matched_keywords = []
        
        for i, word in enumerate(words):
            # Check for negation
            negated = any(neg in words[max(0, i-3):i] for neg in self.negation_words)
            
            # Check for amplifiers
            amplifier = 1.0
            if i > 0 and words[i-1] in self.amplifiers:
                amplifier = self.amplifiers[words[i-1]]
            
            # Score bullish keywords
            if word in self.bullish_keywords:
                score = self.bullish_keywords[word] * amplifier
                if negated:
                    bearish_score += abs(score) * 0.8
                else:
                    bullish_score += score
                matched_keywords.append(word)
            
            # Score bearish keywords
            elif word in self.bearish_keywords:
                score = self.bearish_keywords[word] * amplifier
                if negated:
                    bullish_score += abs(score) * 0.8
                else:
                    bearish_score += abs(score)
                matched_keywords.append(word)
        
        # Check for two-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if phrase in self.bullish_keywords:
                bullish_score += self.bullish_keywords[phrase]
                matched_keywords.append(phrase)
            elif phrase in self.bearish_keywords:
                bearish_score += abs(self.bearish_keywords[phrase])
                matched_keywords.append(phrase)
        
        # Calculate net sentiment
        net_score = bullish_score - bearish_score
        total_score = bullish_score + bearish_score
        
        # Determine sentiment and normalize score
        if total_score == 0:
            sentiment = 'neutral'
            normalized_score = 0.0
            confidence = 0.0
        else:
            # Normalize to -1 to 1 range
            normalized_score = max(-1.0, min(1.0, net_score / 10))
            
            # Calculate confidence based on keyword matches
            confidence = min(1.0, total_score / 20)
            
            # Determine sentiment label
            if normalized_score > 0.2:
                sentiment = 'bullish'
            elif normalized_score < -0.2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
        
        logger.debug(f"Sentiment: {sentiment} (score: {normalized_score:.2f}, confidence: {confidence:.2f})")
        
        return SentimentScore(
            sentiment=sentiment,
            score=normalized_score,
            confidence=confidence,
            keywords=list(set(matched_keywords))[:10]  # Top 10 unique keywords
        )
    
    def is_high_impact(self, text: str, threshold: float = 0.5) -> bool:
        """
        Quick check if text is high-impact and worth deep analysis
        
        Args:
            text: Financial text to analyze
            threshold: Confidence threshold for high-impact classification
            
        Returns:
            True if text appears to be high-impact news
        """
        result = self.analyze(text)
        
        # High impact if strong sentiment with good confidence
        is_high_impact = (
            abs(result.score) > 0.5 and result.confidence > threshold
        ) or any(
            keyword in text.lower() 
            for keyword in ['fed', 'fomc', 'earnings', 'guidance', 'merger', 'acquisition', 'bankruptcy']
        )
        
        if is_high_impact:
            logger.info(f"High-impact event detected: {result.sentiment} (confidence: {result.confidence:.2f})")
        
        return is_high_impact


class NewsFilter:
    """Filter and prioritize news for analysis"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = [s.upper() for s in symbols]
        self.analyzer = SimpleSentimentAnalyzer()
        
    def is_relevant(self, text: str, title: str = "") -> Tuple[bool, List[str]]:
        """
        Check if news is relevant to tracked symbols
        
        Args:
            text: News article text
            title: Article title
            
        Returns:
            Tuple of (is_relevant, matched_symbols)
        """
        full_text = f"{title} {text}".upper()
        
        # Check for symbol mentions
        matched_symbols = []
        for symbol in self.symbols:
            # Look for symbol with word boundaries
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, full_text):
                matched_symbols.append(symbol)
        
        # Check for market-wide events
        market_keywords = ['S&P', 'NASDAQ', 'DOW', 'MARKET', 'FED', 'FOMC', 'ECONOMY']
        is_market_wide = any(keyword in full_text for keyword in market_keywords)
        
        if is_market_wide and 'SPY' in self.symbols:
            matched_symbols.append('SPY')
        
        is_relevant = len(matched_symbols) > 0
        
        return is_relevant, matched_symbols
    
    def prioritize_news(self, news_items: List[Dict]) -> List[Dict]:
        """
        Prioritize news items for analysis
        
        Args:
            news_items: List of news items with 'text' and 'title' fields
            
        Returns:
            Sorted list with priority scores
        """
        prioritized = []
        
        for item in news_items:
            text = item.get('text', '')
            title = item.get('title', '')
            
            # Check relevance
            is_relevant, symbols = self.is_relevant(text, title)
            if not is_relevant:
                continue
            
            # Get sentiment
            sentiment = self.analyzer.analyze(f"{title} {text}")
            
            # Calculate priority score
            priority = sentiment.confidence * abs(sentiment.score)
            
            # Boost for certain keywords
            boost_keywords = ['earnings', 'guidance', 'fed', 'merger', 'acquisition']
            if any(kw in text.lower() for kw in boost_keywords):
                priority *= 1.5
            
            item['priority'] = priority
            item['sentiment'] = sentiment
            item['symbols'] = symbols
            
            prioritized.append(item)
        
        # Sort by priority (highest first)
        prioritized.sort(key=lambda x: x['priority'], reverse=True)
        
        return prioritized


# Future: FinBERT implementation when transformers is installed
class FinBERT:
    """
    FinBERT implementation placeholder.
    Requires: pip install transformers torch
    Model will auto-download on first use (~400MB)
    """
    
    def __init__(self):
        logger.warning("FinBERT not available. Install transformers and torch for advanced sentiment analysis.")
        self.fallback = SimpleSentimentAnalyzer()
    
    def analyze(self, text: str) -> SentimentScore:
        """Fallback to simple analyzer"""
        return self.fallback.analyze(text)