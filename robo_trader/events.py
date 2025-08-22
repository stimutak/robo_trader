"""
Event-driven trading framework.

Processes market events (news, price changes, signals) and generates trading decisions.
Events flow: News → Analysis → Signal → Risk Check → Order
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from collections import deque
import json

from robo_trader.logger import get_logger
from robo_trader.news import NewsItem, NewsAggregator
from robo_trader.intelligence import ClaudeTrader
from robo_trader.risk import RiskManager

logger = get_logger(__name__)


class EventType(Enum):
    """Types of trading events."""
    NEWS = "news"
    PRICE_UPDATE = "price_update"
    SIGNAL = "signal"
    ORDER = "order"
    EXECUTION = "execution"
    RISK_ALERT = "risk_alert"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"


class SignalStrength(Enum):
    """Trading signal strength levels."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class Event:
    """Base event class."""
    event_type: EventType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    priority: int = 0  # Higher = more important


class NewsEvent(Event):
    """News event with analysis requirements."""
    
    def __init__(self, news_item: NewsItem, requires_analysis: bool = True):
        super().__init__(event_type=EventType.NEWS)
        self.news_item = news_item
        self.requires_analysis = requires_analysis
        self.data = {
            "title": self.news_item.title,
            "symbols": self.news_item.symbols,
            "relevance": self.news_item.relevance_score,
            "sentiment": self.news_item.sentiment_score,
        }
        # Higher relevance = higher priority
        self.priority = int(self.news_item.relevance_score * 10)


class SignalEvent(Event):
    """Trading signal generated from analysis."""
    
    def __init__(self, symbol: str, signal: SignalStrength, conviction: float, 
                 reasoning: str, source_events: List[Event] = None):
        super().__init__(event_type=EventType.SIGNAL)
        self.symbol = symbol
        self.signal = signal
        self.conviction = conviction  # 0.0 to 1.0
        self.reasoning = reasoning
        self.source_events = source_events or []
        self.data = {
            "symbol": self.symbol,
            "signal": self.signal.value,
            "conviction": self.conviction,
            "reasoning": self.reasoning,
        }
        # Higher conviction = higher priority
        self.priority = int(self.conviction * 10)


class OrderEvent(Event):
    """Order to be executed."""
    
    def __init__(self, symbol: str, action: str, quantity: int, 
                 order_type: str = "MARKET", limit_price: Optional[float] = None):
        super().__init__(event_type=EventType.ORDER)
        self.symbol = symbol
        self.action = action  # "BUY" or "SELL"
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.data = {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
        }
        self.priority = 8  # Orders are high priority


class EventQueue:
    """Priority queue for events."""
    
    def __init__(self, max_size: int = 1000):
        self.queue: deque = deque(maxlen=max_size)
        self.processed_count = 0
        
    def push(self, event: Event):
        """Add event to queue (sorted by priority)."""
        # Simple insertion sort for small queue
        inserted = False
        for i in range(len(self.queue)):
            if event.priority > self.queue[i].priority:
                self.queue.insert(i, event)
                inserted = True
                break
        if not inserted:
            self.queue.append(event)
            
    def pop(self) -> Optional[Event]:
        """Get highest priority event."""
        if self.queue:
            self.processed_count += 1
            return self.queue.popleft()
        return None
    
    def peek(self) -> Optional[Event]:
        """View highest priority event without removing."""
        return self.queue[0] if self.queue else None
    
    def size(self) -> int:
        """Get queue size."""
        return len(self.queue)
    
    def clear(self):
        """Clear all events."""
        self.queue.clear()


class EventProcessor:
    """Processes events and coordinates trading decisions."""
    
    def __init__(
        self,
        symbols: List[str],
        risk_manager: RiskManager,
        ai_trader: Optional[ClaudeTrader] = None,
        news_aggregator: Optional[NewsAggregator] = None
    ):
        """
        Initialize event processor.
        
        Args:
            symbols: List of symbols to trade
            risk_manager: Risk management instance
            ai_trader: AI analysis instance
            news_aggregator: News feed instance
        """
        self.symbols = symbols
        self.risk_manager = risk_manager
        self.ai_trader = ai_trader
        self.news_aggregator = news_aggregator
        
        self.event_queue = EventQueue()
        self.handlers: Dict[EventType, List[Callable]] = {
            event_type: [] for event_type in EventType
        }
        
        # Register default handlers
        self._register_default_handlers()
        
        # Event history for analysis
        self.event_history: deque = deque(maxlen=1000)
        
        # Signal generation state
        self.last_signals: Dict[str, SignalEvent] = {}
        self.signal_cooldown: Dict[str, datetime] = {}
        
    def _register_default_handlers(self):
        """Register built-in event handlers."""
        self.register_handler(EventType.NEWS, self._handle_news)
        self.register_handler(EventType.SIGNAL, self._handle_signal)
        self.register_handler(EventType.ORDER, self._handle_order)
        
    def register_handler(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type."""
        self.handlers[event_type].append(handler)
        
    async def _handle_news(self, event: NewsEvent):
        """Process news events and generate signals if needed."""
        if not event.requires_analysis:
            logger.debug(f"Skipping analysis for low-relevance news: {event.news_item.title[:50]}")
            return
            
        if not self.ai_trader:
            logger.warning("No AI trader configured, cannot analyze news")
            return
            
        # Check if news is high-impact enough
        if event.news_item.relevance_score < 0.4:
            return
        
        # Deduplicate similar Fed/Powell news within 30 minutes
        title_lower = event.news_item.title.lower()
        is_fed_news = any(keyword in title_lower for keyword in ['powell', 'fed', 'fomc', 'rate cut', 'jackson hole'])
        
        if is_fed_news:
            # Check if we've recently analyzed Fed news
            if hasattr(self, '_last_fed_analysis'):
                time_since = datetime.now(timezone.utc) - self._last_fed_analysis
                if time_since.total_seconds() < 1800:  # 30 minutes
                    logger.debug(f"Skipping duplicate Fed news (analyzed {time_since.total_seconds():.0f}s ago): {event.news_item.title[:50]}")
                    return
            self._last_fed_analysis = datetime.now(timezone.utc)
            
        # Rate limit API calls - max 1 per 10 seconds
        if hasattr(self, '_last_api_call'):
            time_since = datetime.now(timezone.utc) - self._last_api_call
            if time_since.total_seconds() < 10:
                wait_time = 10 - time_since.total_seconds()
                logger.debug(f"Rate limiting: waiting {wait_time:.1f}s before API call")
                await asyncio.sleep(wait_time)
        self._last_api_call = datetime.now(timezone.utc)
        
        logger.info(f"Analyzing high-impact news: {event.news_item.title[:80]}...")
        
        # Prepare context for AI
        market_context = {
            "news_title": event.news_item.title,
            "news_summary": event.news_item.summary,
            "symbols": event.news_item.symbols,
            "sentiment": event.news_item.sentiment_score,
            "source": event.news_item.source,
        }
        
        # Get AI analysis
        try:
            # Convert to format expected by ClaudeTrader
            analysis = await self.ai_trader.analyze_market_event(
                event_text=f"{event.news_item.title}\n\n{event.news_item.summary}",
                symbol=event.news_item.symbols[0] if event.news_item.symbols else "SPY",
                market_data=market_context
            )
            
            if analysis and analysis.get("conviction", 0) > 0.6:
                # Generate trading signal
                signal_strength = self._conviction_to_signal(
                    analysis["conviction"],
                    analysis["direction"]
                )
                
                for symbol in event.news_item.symbols[:3]:  # Limit to 3 symbols
                    # Convert conviction to 0-1 scale if it's in percentage
                    conviction_value = analysis["conviction"]
                    if conviction_value > 1:  # Assume it's a percentage
                        conviction_value = conviction_value / 100.0
                    
                    signal_event = SignalEvent(
                        symbol=symbol,
                        signal=signal_strength,
                        conviction=conviction_value,
                        reasoning=analysis.get("reasoning", f"AI analysis of: {event.news_item.title[:100]}"),
                        source_events=[event]
                    )
                    
                    self.event_queue.push(signal_event)
                    logger.info(
                        f"Generated {signal_strength.value} signal for {symbol} "
                        f"(conviction: {conviction_value:.0%})"
                    )
                    
        except Exception as e:
            logger.error(f"Error analyzing news: {e}")
            
    async def _handle_signal(self, event: SignalEvent):
        """Process trading signals and generate orders."""
        symbol = event.symbol
        
        # Check cooldown
        if symbol in self.signal_cooldown:
            time_since = datetime.now(timezone.utc) - self.signal_cooldown[symbol]
            if time_since.total_seconds() < 300:  # 5 minute cooldown
                logger.debug(f"Signal for {symbol} in cooldown period")
                return
                
        # Check if signal is actionable
        if event.signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]:
            action = "BUY"
        elif event.signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]:
            action = "SELL"
        else:
            return  # HOLD signal
            
        # Determine position size based on conviction (Kelly Criterion simplified)
        base_size = 100  # Base shares
        if event.signal in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
            size_multiplier = min(event.conviction * 2, 3.0)
        else:
            size_multiplier = event.conviction
            
        quantity = int(base_size * size_multiplier)
        
        # Create order event
        order_event = OrderEvent(
            symbol=symbol,
            action=action,
            quantity=quantity,
            order_type="MARKET"
        )
        
        self.event_queue.push(order_event)
        self.signal_cooldown[symbol] = datetime.now(timezone.utc)
        
        logger.info(
            f"Generated {action} order for {quantity} shares of {symbol} "
            f"based on {event.signal.value} signal"
        )
        
    async def _handle_order(self, event: OrderEvent):
        """Validate and forward orders for execution."""
        # This will be connected to the execution module
        logger.info(
            f"Order ready for execution: {event.action} {event.quantity} {event.symbol}"
        )
        
    def _conviction_to_signal(self, conviction: float, direction: str) -> SignalStrength:
        """Convert AI conviction and direction to signal strength."""
        if direction.upper() == "BULLISH":
            if conviction >= 0.8:
                return SignalStrength.STRONG_BUY
            elif conviction >= 0.6:
                return SignalStrength.BUY
        elif direction.upper() == "BEARISH":
            if conviction >= 0.8:
                return SignalStrength.STRONG_SELL
            elif conviction >= 0.6:
                return SignalStrength.SELL
                
        return SignalStrength.HOLD
        
    async def process_events(self):
        """Main event processing loop."""
        while True:
            event = self.event_queue.pop()
            
            if event:
                # Log event
                logger.debug(f"Processing {event.event_type.value} event (priority: {event.priority})")
                
                # Store in history
                self.event_history.append(event)
                
                # Call registered handlers
                for handler in self.handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in handler for {event.event_type}: {e}")
                        
            await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
            
    async def ingest_news(self):
        """Continuously fetch and queue news events."""
        if not self.news_aggregator:
            logger.warning("No news aggregator configured")
            return
            
        while True:
            try:
                # Fetch latest news
                await self.news_aggregator.update()
                
                # Queue high-impact news
                high_impact = self.news_aggregator.get_high_impact_news(min_relevance=0.3)
                
                for news_item in high_impact[:10]:  # Process top 10
                    # Check if we've seen this news
                    if news_item.hash_id not in [e.news_item.hash_id 
                                                  for e in self.event_history 
                                                  if isinstance(e, NewsEvent)]:
                        news_event = NewsEvent(
                            news_item=news_item,
                            requires_analysis=news_item.relevance_score >= 0.4
                        )
                        self.event_queue.push(news_event)
                        
                logger.info(f"Queued {len(high_impact)} news items for processing")
                
            except Exception as e:
                logger.error(f"Error ingesting news: {e}")
                
            # Check for news every 5 minutes
            await asyncio.sleep(300)
            
    def get_event_stats(self) -> Dict[str, Any]:
        """Get statistics about processed events."""
        stats = {
            "queue_size": self.event_queue.size(),
            "processed_total": self.event_queue.processed_count,
            "history_size": len(self.event_history),
            "active_signals": len(self.last_signals),
            "events_by_type": {}
        }
        
        # Count events by type in history
        for event in self.event_history:
            event_type = event.event_type.value
            stats["events_by_type"][event_type] = stats["events_by_type"].get(event_type, 0) + 1
            
        return stats
        
    async def start(self):
        """Start the event processor."""
        logger.info("Starting event processor...")
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self.process_events()),
        ]
        
        if self.news_aggregator:
            tasks.append(asyncio.create_task(self.ingest_news()))
            
        await asyncio.gather(*tasks)


async def main():
    """Test the event framework."""
    
    # Setup components
    symbols = ["AAPL", "TSLA", "NVDA"]
    # Simple risk manager mock for testing
    class MockRiskManager:
        pass
    risk_manager = MockRiskManager()
    
    # Create event processor
    processor = EventProcessor(
        symbols=symbols,
        risk_manager=risk_manager
    )
    
    # Create some test events
    test_news = NewsItem(
        title="Apple Reports Record iPhone Sales",
        summary="Apple exceeded expectations with Q4 iPhone sales...",
        url="https://example.com",
        source="test",
        published=datetime.now(timezone.utc),
        symbols=["AAPL"],
        relevance_score=0.8,
        sentiment_score=0.7
    )
    
    news_event = NewsEvent(news_item=test_news)
    processor.event_queue.push(news_event)
    
    # Test signal
    signal_event = SignalEvent(
        symbol="AAPL",
        signal=SignalStrength.BUY,
        conviction=0.75,
        reasoning="Strong earnings beat"
    )
    processor.event_queue.push(signal_event)
    
    # Process a few events
    for _ in range(5):
        event = processor.event_queue.pop()
        if event:
            print(f"Processing: {event.event_type.value} - Priority: {event.priority}")
            
    print(f"\nEvent stats: {json.dumps(processor.get_event_stats(), indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())