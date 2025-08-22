"""
AI-powered trading runner with news analysis and event processing.
Integrates news feeds, Claude AI, and Kelly position sizing.
"""

import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timezone
import os

from dotenv import load_dotenv
load_dotenv()

from .config import load_config
from .execution import Order, PaperExecutor
from .ibkr_client import IBKRClient
from .risk import RiskManager, Position
from .logger import get_logger
from .portfolio import Portfolio
from .retry import retry_async

# AI components
from .news import NewsAggregator
from .intelligence import ClaudeTrader
from .events import EventProcessor, EventType, SignalEvent, OrderEvent
from .kelly import KellyCalculator
from .options_flow import OptionsFlowAnalyzer
from .database import TradingDatabase

logger = get_logger(__name__)


class AITradingSystem:
    """Complete AI-powered trading system."""
    
    def __init__(
        self,
        symbols: List[str],
        use_ai: bool = True,
        news_check_interval: int = 300,  # 5 minutes
        capital: float = 100000.0
    ):
        """
        Initialize AI trading system.
        
        Args:
            symbols: List of symbols to trade
            use_ai: Whether to use Claude AI for analysis
            news_check_interval: Seconds between news checks
            capital: Starting capital
        """
        self.symbols = symbols
        self.use_ai = use_ai
        self.news_check_interval = news_check_interval
        self.capital = capital
        
        # Core components (will be initialized in setup)
        self.ib_client: Optional[IBKRClient] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[PaperExecutor] = None
        self.portfolio: Optional[Portfolio] = None
        
        # AI components
        self.news_aggregator: Optional[NewsAggregator] = None
        self.ai_trader: Optional[ClaudeTrader] = None
        self.event_processor: Optional[EventProcessor] = None
        self.kelly_calc: Optional[KellyCalculator] = None
        self.options_flow: Optional[OptionsFlowAnalyzer] = None
        
        # Database for persistence
        self.db: Optional[TradingDatabase] = None
        
        # State tracking
        self.is_running = False
        self.stats = {
            "news_processed": 0,
            "signals_generated": 0,
            "trades_executed": 0,
            "ai_analyses": 0,
            "options_signals": 0
        }
        
    async def setup(self):
        """Initialize all components."""
        logger.info("Setting up AI Trading System...")
        
        # Load config
        cfg = load_config()
        
        # Setup IB connection
        self.ib_client = IBKRClient(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
        await retry_async(lambda: self.ib_client.connect(readonly=False))
        logger.info("✓ Connected to Interactive Brokers")
        
        # Setup risk and execution
        self.risk_manager = RiskManager(
            max_daily_loss=cfg.max_daily_loss,
            max_position_risk_pct=cfg.max_position_risk_pct,
            max_symbol_exposure_pct=cfg.max_symbol_exposure_pct,
            max_leverage=cfg.max_leverage
        )
        self.executor = PaperExecutor(slippage_bps=5.0)
        self.portfolio = Portfolio(self.capital)
        logger.info("✓ Risk management and execution ready")
        
        # Initialize database
        self.db = TradingDatabase()
        logger.info("✓ Database initialized for data persistence")
        
        # Setup AI components
        self.news_aggregator = NewsAggregator(self.symbols, lookback_hours=6)
        logger.info("✓ News aggregator initialized")
        
        if self.use_ai and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.ai_trader = ClaudeTrader()
                logger.info("✓ Claude AI trader initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Claude AI: {e}")
                self.use_ai = False
        
        self.kelly_calc = KellyCalculator(capital=self.capital)
        logger.info("✓ Kelly calculator ready")
        
        # Setup options flow analyzer
        self.options_flow = OptionsFlowAnalyzer(self.ib_client)
        logger.info("✓ Options flow analyzer initialized")
        
        # Setup event processor
        self.event_processor = EventProcessor(
            symbols=self.symbols,
            risk_manager=self.risk_manager,
            ai_trader=self.ai_trader,
            news_aggregator=self.news_aggregator
        )
        
        # Register custom handlers
        self.event_processor.register_handler(EventType.ORDER, self._handle_order)
        logger.info("✓ Event processor configured")
        
        logger.info("AI Trading System setup complete!")
        
    async def _handle_order(self, event: OrderEvent):
        """Execute orders through IB."""
        try:
            # Get current price
            bars = await self.ib_client.fetch_recent_bars(
                symbol=event.symbol,
                duration="1 D",
                bar_size="1 min"
            )
            
            if bars.empty:
                logger.warning(f"No price data for {event.symbol}")
                return
                
            current_price = bars.iloc[-1]['close']
            
            # Create order
            order = Order(
                symbol=event.symbol,
                action=event.action,
                quantity=event.quantity,
                order_type=event.order_type,
                limit_price=event.limit_price
            )
            
            # Risk check
            position = Position(
                symbol=event.symbol,
                quantity=event.quantity if event.action == "BUY" else -event.quantity,
                entry_price=current_price
            )
            
            if self.risk_manager.check_position_risk(position, self.portfolio.equity):
                # Execute
                fill = self.executor.execute(order, current_price)
                if fill:
                    self.portfolio.update_position(
                        event.symbol,
                        event.quantity if event.action == "BUY" else -event.quantity,
                        fill.fill_price
                    )
                    
                    logger.info(
                        f"✓ Executed: {event.action} {event.quantity} {event.symbol} "
                        f"@ ${fill.fill_price:.2f}"
                    )
                    self.stats["trades_executed"] += 1
                    
                    # Save trade to database
                    if self.db:
                        trade_data = {
                            'symbol': event.symbol,
                            'action': event.action,
                            'quantity': event.quantity,
                            'price': fill.fill_price,
                            'notional': fill.fill_price * event.quantity,
                            'ai_confidence': getattr(event, 'conviction', None),
                            'ai_reasoning': getattr(event, 'reasoning', None),
                            'strategy': 'AI_EVENT_DRIVEN',
                            'commission': 1.0  # Estimate
                        }
                        self.db.save_trade(trade_data)
            else:
                logger.warning(f"Risk check failed for {event.symbol} order")
                
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            
    async def process_news_cycle(self):
        """Fetch and process news in a cycle."""
        while self.is_running:
            try:
                # Fetch latest news
                logger.info("Fetching latest news...")
                new_count, total = await self.news_aggregator.update()
                
                if new_count > 0:
                    logger.info(f"Found {new_count} new articles")
                    self.stats["news_processed"] += new_count
                    
                    # Process high-impact news through events
                    high_impact = self.news_aggregator.get_high_impact_news(min_relevance=0.4)
                    
                    # Push news to dashboard
                    await self._push_news_to_dashboard(high_impact)
                    
                    for news_item in high_impact[:5]:  # Process top 5
                        # The event processor will handle AI analysis
                        pass  # Events are processed automatically
                        
                # Wait before next check
                await asyncio.sleep(self.news_check_interval)
                
            except Exception as e:
                logger.error(f"Error in news cycle: {e}")
                await asyncio.sleep(60)  # Wait a bit on error
                
    async def process_market_data(self):
        """Monitor market data and generate signals."""
        await asyncio.sleep(5)  # Initial delay to avoid event loop conflict
        while self.is_running:
            try:
                # Get latest prices for all symbols
                for symbol in self.symbols:
                    try:
                        bars = await self.ib_client.fetch_recent_bars(
                            symbol=symbol,
                            duration="1 D",
                            bar_size="5 mins"
                        )
                        
                        if not bars.empty:
                            current_price = bars.iloc[-1]['close']
                            prev_close = bars.iloc[-2]['close'] if len(bars) > 1 else current_price
                            
                            # Calculate basic metrics
                            price_change = (current_price - prev_close) / prev_close
                            
                            # Log significant moves
                            if abs(price_change) > 0.01:  # 1% move
                                logger.info(f"{symbol}: ${current_price:.2f} ({price_change:+.2%})")
                    except Exception as e:
                        logger.debug(f"Could not fetch data for {symbol}: {e}")
                            
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.debug(f"Market data cycle error: {e}")
                await asyncio.sleep(30)
                
    async def process_options_flow(self):
        """Monitor options flow for unusual activity."""
        while self.is_running:
            try:
                # Scan options every 5 minutes
                logger.info("Scanning options flow for unusual activity...")
                signals = await self.options_flow.scan_options_flow(self.symbols)
                
                if signals:
                    # Get summary
                    summary = self.options_flow.get_flow_summary(signals)
                    logger.info(
                        f"Options flow: {summary['total_signals']} signals, "
                        f"Bullish: {summary['bullish_flow']}, "
                        f"Bearish: {summary['bearish_flow']}"
                    )
                    self.stats["options_signals"] += len(signals)
                    
                    # Save options signals to database
                    if self.db:
                        for signal in signals:
                            signal_data = {
                                'symbol': signal.symbol,
                                'strike': signal.strike,
                                'expiry': signal.expiry,
                                'option_type': signal.option_type,
                                'signal_type': signal.signal_type,
                                'volume': signal.volume,
                                'open_interest': signal.open_interest,
                                'confidence': signal.confidence,
                                'premium': signal.premium,
                                'implied_volatility': getattr(signal, 'implied_volatility', None)
                            }
                            self.db.save_options_signal(signal_data)
                    
                    # Send to dashboard
                    await self._push_options_to_dashboard(signals)
                    
                    # Process high-confidence signals through AI
                    for signal in signals[:3]:  # Top 3 signals
                        if signal.confidence >= 70:
                            # Create synthetic news item for AI analysis
                            options_news = {
                                'title': f"OPTIONS ALERT: {signal.interpretation}",
                                'summary': (
                                    f"{signal.symbol} {signal.option_type} "
                                    f"Strike: ${signal.strike} Expiry: {signal.expiry} "
                                    f"Volume: {signal.volume} (Volume/OI: {signal.volume_oi_ratio:.1f}x) "
                                    f"Premium: ${signal.premium:,.0f} "
                                    f"Signal: {signal.signal_type.upper()}"
                                ),
                                'source': 'options_flow',
                                'symbols': [signal.symbol],
                                'relevance': signal.confidence / 100
                            }
                            
                            # Have AI analyze this options activity
                            if self.ai_trader:
                                analysis = await self.ai_trader.analyze_event(
                                    signal.symbol,
                                    options_news['title'],
                                    options_news['summary']
                                )
                                
                                # Save AI decision to database
                                if analysis and self.db:
                                    decision_data = {
                                        'event_type': 'OPTIONS_FLOW',
                                        'event_data': {
                                            'symbol': signal.symbol,
                                            'signal_type': signal.signal_type,
                                            'strike': signal.strike,
                                            'expiry': str(signal.expiry),
                                            'volume': signal.volume,
                                            'premium': signal.premium
                                        },
                                        'decision': analysis.get('direction', 'HOLD').upper(),
                                        'confidence': analysis.get('conviction', 0),
                                        'reasoning': analysis.get('reasoning', '')
                                    }
                                    self.db.save_ai_decision(decision_data)
                                
                                if analysis and analysis['conviction'] >= 75:
                                    logger.info(
                                        f"HIGH CONVICTION OPTIONS SIGNAL: "
                                        f"{signal.symbol} {analysis['direction']} "
                                        f"({analysis['conviction']}% confidence)"
                                    )
                                    self.stats["ai_analyses"] += 1
                
                # Wait 5 minutes before next scan
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error processing options flow: {e}")
                await asyncio.sleep(60)
                
    async def save_pnl_snapshot(self):
        """Save P&L snapshot to database every 10 minutes."""
        await asyncio.sleep(10)  # Initial delay
        while self.is_running:
            try:
                if self.db and self.portfolio:
                    # Get current P&L data
                    pnl_data = {
                        'total_pnl': self.portfolio.total_pnl,
                        'daily_pnl': self.portfolio.daily_pnl,
                        'realized_pnl': self.portfolio.realized_pnl,
                        'unrealized_pnl': self.portfolio.unrealized_pnl,
                        'positions_count': len(self.portfolio.positions),
                        'positions': [
                            {
                                'symbol': symbol,
                                'quantity': pos.quantity,
                                'entry_price': pos.entry_price,
                                'current_price': pos.current_price,
                                'pnl': pos.unrealized_pnl
                            }
                            for symbol, pos in self.portfolio.positions.items()
                        ]
                    }
                    self.db.save_pnl_snapshot(pnl_data)
                    logger.debug(f"Saved P&L snapshot: ${pnl_data['total_pnl']:.2f}")
                
                await asyncio.sleep(600)  # Save every 10 minutes
            except Exception as e:
                logger.error(f"Error saving P&L snapshot: {e}")
                await asyncio.sleep(60)
    
    async def run(self):
        """Main trading loop."""
        if not self.is_running:
            self.is_running = True
            logger.info("Starting AI Trading System...")
            
            # Start all async tasks
            tasks = [
                asyncio.create_task(self.event_processor.process_events()),
                asyncio.create_task(self.event_processor.ingest_news()),
                asyncio.create_task(self.process_market_data()),
                asyncio.create_task(self.save_pnl_snapshot()),  # Add P&L tracking
            ]
            
            if self.use_ai:
                tasks.append(asyncio.create_task(self.process_news_cycle()))
                tasks.append(asyncio.create_task(self.process_options_flow()))
                
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info("Shutting down AI Trading System...")
                self.is_running = False
                
    async def _push_news_to_dashboard(self, news_items):
        """Push news to dashboard for ticker display."""
        try:
            import aiohttp
            news_data = []
            for item in news_items[:20]:  # Top 20 for ticker
                news_data.append({
                    'title': item.title[:100] if hasattr(item, 'title') else str(item).split('-')[0][:100],
                    'source': item.source if hasattr(item, 'source') else 'Unknown',
                    'sentiment': item.sentiment_score if hasattr(item, 'sentiment_score') else 0.0,
                    'time': item.published.strftime("%H:%M") if hasattr(item, 'published') else datetime.now().strftime("%H:%M"),
                    'url': item.url if hasattr(item, 'url') else None
                })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:5555/api/news',
                    json={'news': news_data},
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Successfully pushed {len(news_data)} news items to dashboard")
                    else:
                        logger.warning(f"Dashboard returned status {resp.status}")
        except Exception as e:
            logger.warning(f"Could not push news to dashboard: {e}")
    
    async def _push_options_to_dashboard(self, signals):
        """Push options flow signals to dashboard."""
        try:
            import aiohttp
            from dataclasses import asdict
            
            # Convert signals to dict format
            options_data = []
            for signal in signals[:10]:  # Top 10 signals
                signal_dict = asdict(signal)
                # Convert datetime to string
                signal_dict['timestamp'] = signal.timestamp.strftime("%H:%M")
                signal_dict['time'] = signal.timestamp.strftime("%H:%M")
                options_data.append(signal_dict)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'http://localhost:5555/api/options',
                    json={'options': options_data},
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"Successfully pushed {len(options_data)} options signals to dashboard")
                    else:
                        logger.warning(f"Dashboard returned status {resp.status} for options")
        except Exception as e:
            logger.warning(f"Could not push options to dashboard: {e}")
    
    async def stop(self):
        """Stop the trading system."""
        self.is_running = False
        if self.ib_client and self.ib_client.ib:
            self.ib_client.ib.disconnect()
        logger.info("AI Trading System stopped")
        
    def get_status(self) -> Dict:
        """Get system status for dashboard."""
        return {
            "is_running": self.is_running,
            "ai_enabled": self.use_ai and self.ai_trader is not None,
            "symbols": self.symbols,
            "stats": self.stats,
            "portfolio": {
                "equity": self.portfolio.equity if self.portfolio else 0,
                "cash": self.portfolio.cash if self.portfolio else 0,
                "positions": self.portfolio.positions if self.portfolio else {}
            },
            "event_stats": self.event_processor.get_event_stats() if self.event_processor else {}
        }


async def main():
    """Run the AI trading system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI-Powered Trading System")
    parser.add_argument("--symbols", type=str, default="SPY,QQQ,AAPL,TSLA,NVDA",
                       help="Comma-separated symbols to trade")
    parser.add_argument("--no-ai", action="store_true",
                       help="Disable AI analysis")
    parser.add_argument("--capital", type=float, default=100000,
                       help="Starting capital")
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(",")
    
    # Create and setup system
    system = AITradingSystem(
        symbols=symbols,
        use_ai=not args.no_ai,
        capital=args.capital
    )
    
    await system.setup()
    
    # Run
    try:
        await system.run()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        await system.stop()


if __name__ == "__main__":
    asyncio.run(main())