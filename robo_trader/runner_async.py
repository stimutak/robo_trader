"""
Enhanced async runner with parallel symbol processing and the new async IBKR client.

This implements Phase 1 F4: Enable Parallel Symbol Processing
- Processes multiple symbols concurrently
- Uses the new async IBKR client with connection pooling
- Implements proper async patterns throughout
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .analysis.correlation_integration import (
    AsyncCorrelationManager,
    CorrelationBasedPositionSizer,
)
from .clients import AsyncIBKRClient, ConnectionConfig
from .config import load_config
from .correlation import CorrelationTracker
from .database_async import AsyncTradingDatabase
from .execution import Order, PaperExecutor
from .logger import get_logger
from .market_hours import is_market_open, get_market_session, seconds_until_market_open
from .monitoring.performance import PerformanceMonitor, Timer

# Import WebSocket client for real-time updates
try:
    from .websocket_client import ws_client
    WEBSOCKET_ENABLED = True
except ImportError:
    ws_client = None
    WEBSOCKET_ENABLED = False
from .portfolio import Portfolio
from .risk import Position, RiskManager
from .strategies import sma_crossover_signals

logger = get_logger(__name__)


@dataclass
class SymbolResult:
    """Result from processing a single symbol."""

    symbol: str
    signal: int
    price: float
    quantity: int
    executed: bool
    message: str
    data: Optional[pd.DataFrame] = None


class AsyncRunner:
    """Enhanced async runner with parallel processing capabilities."""

    def __init__(
        self,
        duration: str = "10 D",
        bar_size: str = "30 mins",
        sma_fast: int = 10,
        sma_slow: int = 20,
        slippage_bps: float = 0.0,
        max_order_notional: Optional[float] = None,
        max_daily_notional: Optional[float] = None,
        default_cash: Optional[float] = None,
        max_concurrent_symbols: int = 8,
        use_correlation_sizing: bool = True,
        max_correlation: float = 0.7,
    ):
        self.duration = duration
        self.bar_size = bar_size
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.slippage_bps = slippage_bps
        self.max_order_notional = max_order_notional
        self.max_daily_notional = max_daily_notional
        self.default_cash = default_cash
        self.max_concurrent_symbols = max_concurrent_symbols
        self.use_correlation_sizing = use_correlation_sizing
        self.max_correlation = max_correlation

        # Will be initialized in setup
        self.cfg = None
        self.client = None
        self.db = None
        self.risk = None
        self.executor = None
        self.portfolio = None
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_executed_notional = 0.0
        self.monitor = PerformanceMonitor()
        
        # Correlation components
        self.correlation_tracker = None
        self.position_sizer = None
        self.correlation_manager = None

    async def setup(self):
        """Initialize all components."""
        self.cfg = load_config()

        # Create async IBKR client with connection pooling
        conn_config = ConnectionConfig(
            host=self.cfg.ibkr.host,
            port=self.cfg.ibkr.port,
            client_id=self.cfg.ibkr.client_id,
            readonly=self.cfg.ibkr.readonly,
            max_connections=min(5, self.max_concurrent_symbols),
        )
        self.client = AsyncIBKRClient(conn_config)
        await self.client.connect()

        # Initialize async database
        self.db = AsyncTradingDatabase()
        await self.db.initialize()

        # Initialize risk manager
        self.risk = RiskManager(
            max_daily_loss=self.cfg.risk.max_daily_loss_pct,
            max_position_risk_pct=self.cfg.risk.max_position_pct,
            max_symbol_exposure_pct=self.cfg.risk.max_sector_exposure_pct,
            max_leverage=self.cfg.risk.max_leverage,
            max_order_notional=self.max_order_notional or self.cfg.risk.max_order_notional,
            max_daily_notional=self.max_daily_notional or self.cfg.risk.max_daily_notional,
        )

        # Initialize executor
        self.executor = PaperExecutor(slippage_bps=self.slippage_bps)

        # Initialize portfolio
        starting_cash = (
            self.default_cash if self.default_cash is not None else self.cfg.default_cash
        )
        self.portfolio = Portfolio(starting_cash)

        # Initialize correlation components if enabled
        if self.use_correlation_sizing:
            self.correlation_tracker = CorrelationTracker(
                lookback_days=60,
                correlation_threshold=self.max_correlation
            )
            self.position_sizer = CorrelationBasedPositionSizer(
                correlation_tracker=self.correlation_tracker,
                max_correlation=self.max_correlation,
                correlation_penalty_factor=0.5,
                max_correlated_exposure=0.3
            )
            self.correlation_manager = AsyncCorrelationManager(
                correlation_tracker=self.correlation_tracker,
                position_sizer=self.position_sizer
            )
            await self.correlation_manager.start(update_interval=300)
            logger.info("Correlation-based position sizing enabled")

        logger.info("AsyncRunner setup complete")

    async def teardown(self):
        """Clean up resources."""
        if self.correlation_manager:
            await self.correlation_manager.stop()
        if self.client:
            await self.client.disconnect()
        if self.db:
            await self.db.close()

    async def fetch_and_store_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for a symbol and store in database."""
        # Check if market is open before fetching data
        if not is_market_open():
            session = get_market_session()
            logger.debug(f"Market is {session}, skipping data fetch for {symbol}")
            return None
        
        try:
            with Timer("data_fetch", self.monitor):
                df = await self.client.fetch_recent_bars(
                    symbol, duration=self.duration, bar_size=self.bar_size
                )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Store market data in database
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            
            # Prepare batch data for efficient storage
            batch_data = []
            for timestamp, row in df.iterrows():
                # Convert pandas Timestamp to datetime for SQLite compatibility
                if hasattr(timestamp, 'to_pydatetime'):
                    timestamp = timestamp.to_pydatetime()
                batch_data.append({
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": int(row.get("volume", 0)),
                })
            
            if batch_data:
                with Timer("database_write", self.monitor):
                    await self.db.batch_store_market_data(batch_data)
                await self.monitor.record_data_points(len(batch_data))

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    async def process_symbol(self, symbol: str) -> SymbolResult:
        """Process a single symbol - fetch data, generate signal, execute if needed."""
        # Fetch and store market data
        df = await self.fetch_and_store_data(symbol)
        if df is None or df.empty:
            return SymbolResult(
                symbol=symbol,
                signal=0,
                price=0,
                quantity=0,
                executed=False,
                message="No data available",
            )
        
        # Send real-time price update via WebSocket
        if WEBSOCKET_ENABLED and ws_client and df is not None and not df.empty:
            try:
                latest_price = float(df['close'].iloc[-1])
                ws_client.send_market_update(symbol, latest_price)
                logger.info(f"Sent WebSocket update for {symbol}: ${latest_price:.2f}")
            except Exception as e:
                logger.error(f"Could not send WebSocket update: {e}")

        # Generate trading signal
        with Timer("signal_generation", self.monitor):
            signals = sma_crossover_signals(
                pd.DataFrame({"close": df["close"]}),
                fast=self.sma_fast,
                slow=self.sma_slow,
            )
        
        # Update correlation tracker with price data if enabled
        if self.use_correlation_sizing and self.correlation_tracker:
            # Add price series for correlation calculation
            self.correlation_tracker.add_price_series(
                symbol=symbol,
                prices=df['close'],
                sector=None  # TODO: Add sector classification
            )
        
        last = signals.iloc[-1]
        price = float(last.get("close", df["close"].iloc[-1]))

        # Get current equity
        equity = self.portfolio.equity({symbol: price for symbol in self.positions})

        # Record signal in database
        signal_value = int(last.get("signal", 0))
        if signal_value != 0:
            await self.db.record_signal(
                symbol,
                "SMA_CROSSOVER",
                "BUY" if signal_value == 1 else "SELL",
                abs(signal_value),
            )
            
            # Send signal update via WebSocket
            if WEBSOCKET_ENABLED and ws_client:
                try:
                    signal_type = "BUY" if signal_value == 1 else "SELL"
                    ws_client.send_signal_update(symbol, signal_type, abs(signal_value))
                except Exception as e:
                    logger.debug(f"Could not send signal WebSocket update: {e}")

        # Execute trades based on signal
        executed = False
        message = "No action"
        quantity = 0

        if signal_value == 1:  # Buy signal
            qty = self.risk.position_size(equity, price)
            
            # Apply correlation-based position sizing if enabled
            if self.use_correlation_sizing and self.correlation_manager:
                adjusted_qty, sizing_reason = await self.correlation_manager.get_adjusted_position_size(
                    symbol=symbol,
                    base_size=qty,
                    current_positions=self.positions,
                    portfolio_value=equity
                )
                if adjusted_qty != qty:
                    logger.info(f"Position size adjusted for {symbol}: {qty} -> {adjusted_qty} ({sizing_reason})")
                    qty = adjusted_qty
            
            ok, msg = self.risk.validate_order(
                symbol,
                qty,
                price,
                equity,
                self.daily_pnl,
                self.positions,
                self.daily_executed_notional,
            )

            if ok and qty > 0:
                with Timer("order_execution", self.monitor):
                    res = self.executor.place_order(
                        Order(symbol=symbol, quantity=qty, side="BUY", price=price)
                    )
                if res.ok:
                    fill_price = res.fill_price or price
                    self.positions[symbol] = Position(symbol, qty, fill_price)
                    self.portfolio.update_fill(symbol, "BUY", qty, fill_price)
                    self.daily_executed_notional += price * qty

                    # Record trade in database
                    await self.db.record_trade(
                        symbol,
                        "BUY",
                        qty,
                        fill_price,
                        slippage=(fill_price - price) * qty if res.fill_price else 0,
                    )
                    await self.db.update_position(symbol, qty, fill_price, price)

                    await self.monitor.record_order_placed(symbol, qty)
                    await self.monitor.record_trade_executed(symbol, "BUY", qty)
                    executed = True
                    quantity = qty
                    message = f"Bought {qty} shares at ${fill_price:.2f}"
                    
                    # Send trade update via WebSocket
                    if WEBSOCKET_ENABLED and ws_client:
                        try:
                            ws_client.send_trade_update(symbol, "BUY", qty, fill_price)
                        except Exception as e:
                            logger.debug(f"Could not send trade WebSocket update: {e}")
                else:
                    message = f"Buy order failed: {res.msg}"
            else:
                message = f"Buy signal rejected: {msg}"

        elif signal_value == -1 and symbol in self.positions:  # Sell signal
            pos = self.positions[symbol]
            with Timer("order_execution", self.monitor):
                res = self.executor.place_order(
                    Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=price)
                )
            if res.ok:
                fill_price = res.fill_price or price
                self.portfolio.update_fill(symbol, "SELL", pos.quantity, fill_price)
                self.daily_pnl = self.portfolio.realized_pnl

                # Record trade in database
                await self.db.record_trade(
                    symbol,
                    "SELL",
                    pos.quantity,
                    fill_price,
                    slippage=(fill_price - price) * pos.quantity if res.fill_price else 0,
                )
                await self.db.update_position(symbol, 0, 0, 0)  # Close position

                del self.positions[symbol]
                await self.monitor.record_order_placed(symbol, pos.quantity)
                await self.monitor.record_trade_executed(symbol, "SELL", pos.quantity)
                executed = True
                quantity = pos.quantity
                message = f"Sold {pos.quantity} shares at ${fill_price:.2f}"
            else:
                message = f"Sell order failed: {res.msg}"

        return SymbolResult(
            symbol=symbol,
            signal=signal_value,
            price=price,
            quantity=quantity,
            executed=executed,
            message=message,
            data=df,
        )

    async def run_parallel(self, symbols: List[str]) -> List[SymbolResult]:
        """Process multiple symbols in parallel with concurrency control."""
        semaphore = asyncio.Semaphore(self.max_concurrent_symbols)

        async def process_with_semaphore(symbol: str) -> SymbolResult:
            async with semaphore:
                result = await self.process_symbol(symbol)
                await self.monitor.record_symbol_processed(
                    symbol, success=result.executed or result.signal == 0
                )
                return result

        # Process all symbols concurrently
        tasks = [process_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        # Log results
        for result in results:
            if isinstance(result, SymbolResult):
                logger.info(
                    f"{result.symbol}: Signal={result.signal}, "
                    f"Price=${result.price:.2f}, {result.message}"
                )
            else:
                logger.error(f"Error processing symbol: {result}")

        return [r for r in results if isinstance(r, SymbolResult)]

    async def update_account_summary(self):
        """Update account summary in database."""
        market_prices = {symbol: pos.avg_price for symbol, pos in self.positions.items()}
        equity = self.portfolio.equity(market_prices)
        unrealized = self.portfolio.compute_unrealized(market_prices)

        await self.db.update_account(
            cash=self.portfolio.cash,
            equity=equity,
            daily_pnl=self.daily_pnl,
            realized_pnl=self.portfolio.realized_pnl,
            unrealized_pnl=unrealized,
        )

        logger.info(
            f"Trading cycle complete. Equity: ${equity:,.2f}, "
            f"Daily P&L: ${self.daily_pnl:,.2f}, "
            f"Positions: {len(self.positions)}"
        )

    async def run(self, symbols: Optional[List[str]] = None):
        """Main run method - process all symbols and update account."""
        await self.setup()
        try:
            # Check market status
            if not is_market_open():
                session = get_market_session()
                seconds_to_open = seconds_until_market_open()
                hours_to_open = seconds_to_open / 3600
                logger.warning(
                    f"Market is currently {session}. "
                    f"Next market open in {hours_to_open:.1f} hours. "
                    f"Skipping data collection for equities."
                )
                # Still process symbols but won't fetch new data
            else:
                logger.info(f"Market is open, proceeding with data collection")
            
            symbols_to_process = symbols if symbols else self.cfg.symbols
            logger.info(
                f"Processing {len(symbols_to_process)} symbols "
                f"with max {self.max_concurrent_symbols} concurrent"
            )

            # Process symbols in parallel
            results = await self.run_parallel(symbols_to_process)

            # Update account summary
            await self.update_account_summary()

            # Log execution summary
            executed_count = sum(1 for r in results if r.executed)
            logger.info(
                f"Processed {len(results)} symbols, "
                f"executed {executed_count} trades"
            )
            
            # Log correlation metrics if enabled
            if self.use_correlation_sizing and self.position_sizer:
                corr_metrics = self.position_sizer.get_metrics()
                logger.info(
                    f"Correlation metrics: positions_reduced={corr_metrics['positions_reduced']}, "
                    f"positions_rejected={corr_metrics['positions_rejected']}, "
                    f"avg_correlation={corr_metrics['avg_correlation']:.3f}"
                )
                
                # Log high correlation pairs
                high_corr_pairs = self.position_sizer.get_high_correlation_pairs()
                if high_corr_pairs:
                    logger.warning(f"High correlation pairs: {high_corr_pairs[:3]}")
            
            # Log performance metrics
            await self.monitor.log_performance_summary()

        finally:
            await self.teardown()


async def run_once(
    symbols: Optional[List[str]] = None,
    duration: str = "10 D",
    bar_size: str = "30 mins",
    sma_fast: int = 10,
    sma_slow: int = 20,
    slippage_bps: float = 0.0,
    max_order_notional: Optional[float] = None,
    max_daily_notional: Optional[float] = None,
    default_cash: Optional[float] = None,
    max_concurrent: int = 8,
) -> None:
    """Run the trading system once with parallel processing."""
    runner = AsyncRunner(
        duration=duration,
        bar_size=bar_size,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
        slippage_bps=slippage_bps,
        max_order_notional=max_order_notional,
        max_daily_notional=max_daily_notional,
        default_cash=default_cash,
        max_concurrent_symbols=max_concurrent,
        use_correlation_sizing=True,  # FIXED: Enabled M5 correlation integration
    )
    await runner.run(symbols)


async def run_continuous(
    symbols: Optional[List[str]] = None,
    duration: str = "10 D",
    bar_size: str = "30 mins",
    sma_fast: int = 10,
    sma_slow: int = 20,
    slippage_bps: float = 0.0,
    max_order_notional: Optional[float] = None,
    max_daily_notional: Optional[float] = None,
    default_cash: Optional[float] = None,
    max_concurrent: int = 8,
    interval_seconds: int = 300,
) -> None:
    """Run the trading system continuously with market hours checking."""
    import signal
    import pytz
    from datetime import datetime
    
    # Setup signal handling for graceful shutdown
    shutdown_flag = False
    
    def signal_handler(signum, frame):
        nonlocal shutdown_flag
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_flag = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting continuous trading system...")
    logger.info("Press Ctrl+C to stop gracefully")
    
    while not shutdown_flag:
        eastern = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern)
        
        # Check market status but still run (data might be useful even after hours)
        if not is_market_open():
            session = get_market_session()
            seconds_to_open = seconds_until_market_open()
            
            # During extended hours or shortly before open, run more frequently
            if session in ["after-hours", "pre-market"] or seconds_to_open < 3600:
                wait_time = min(interval_seconds, 300)  # Max 5 minutes
            else:
                # Market closed, check less frequently
                wait_time = min(1800, seconds_to_open // 2)  # Max 30 minutes
                logger.info(
                    f"Market {session}. Next open in {seconds_to_open/3600:.1f} hours. "
                    f"Waiting {wait_time/60:.1f} minutes..."
                )
                await asyncio.sleep(wait_time)
                continue
        
        try:
            logger.info(f"Starting trading cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            # Run the trading system
            runner = AsyncRunner(
                duration=duration,
                bar_size=bar_size,
                sma_fast=sma_fast,
                sma_slow=sma_slow,
                slippage_bps=slippage_bps,
                max_order_notional=max_order_notional,
                max_daily_notional=max_daily_notional,
                default_cash=default_cash,
                max_concurrent_symbols=max_concurrent,
                use_correlation_sizing=True,  # FIXED: Enabled M5 correlation integration
            )
            
            await runner.run(symbols)
            
            # Wait before next iteration
            if not shutdown_flag and is_market_open():
                logger.info(f"Waiting {interval_seconds/60:.1f} minutes before next iteration...")
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            if not shutdown_flag:
                logger.info("Waiting 1 minute before retry...")
                await asyncio.sleep(60)
    
    logger.info("Trading system shutdown complete")


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Async Robo Trader with parallel symbol processing"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols to override config",
        default="",
    )
    parser.add_argument(
        "--duration", type=str, help="IB duration string (e.g. '10 D')", default="10 D"
    )
    parser.add_argument(
        "--bar-size", type=str, help="IB bar size (e.g. '30 mins')", default="30 mins"
    )
    parser.add_argument(
        "--confirm-live",
        action="store_true",
        help="Required confirmation flag for live mode",
    )
    parser.add_argument("--sma-fast", type=int, default=10, help="Fast SMA window")
    parser.add_argument("--sma-slow", type=int, default=20, help="Slow SMA window")
    parser.add_argument(
        "--slippage-bps", type=float, default=0.0, help="Paper slippage in basis points"
    )
    parser.add_argument(
        "--max-order-notional",
        type=float,
        default=None,
        help="Per-order notional ceiling",
    )
    parser.add_argument(
        "--max-daily-notional",
        type=float,
        default=None,
        help="Per-day notional ceiling",
    )
    parser.add_argument(
        "--default-cash",
        type=float,
        default=None,
        help="Override starting cash for paper run",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent symbol processing (default: 8)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (default: run continuously)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between trading cycles (default: 300 = 5 minutes)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if cfg.execution.mode == "live" and not args.confirm_live:
        raise SystemExit("Refusing to run in live mode without --confirm-live")

    override_symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else None
    )

    if args.once:
        # Run once and exit (for testing/debugging)
        asyncio.run(
            run_once(
                symbols=override_symbols,
                duration=args.duration,
                bar_size=args.bar_size,
                sma_fast=args.sma_fast,
                sma_slow=args.sma_slow,
                slippage_bps=args.slippage_bps,
                max_order_notional=args.max_order_notional,
                max_daily_notional=args.max_daily_notional,
                default_cash=args.default_cash,
                max_concurrent=args.max_concurrent,
            )
        )
    else:
        # Run continuously (default behavior)
        asyncio.run(
            run_continuous(
                symbols=override_symbols,
                duration=args.duration,
                bar_size=args.bar_size,
                sma_fast=args.sma_fast,
                sma_slow=args.sma_slow,
                slippage_bps=args.slippage_bps,
                max_order_notional=args.max_order_notional,
                max_daily_notional=args.max_daily_notional,
                default_cash=args.default_cash,
                max_concurrent=args.max_concurrent,
                interval_seconds=args.interval,
            )
        )


if __name__ == "__main__":
    main()