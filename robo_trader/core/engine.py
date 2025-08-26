"""
Async trading engine with health monitoring and graceful shutdown.

This module provides the core event-driven trading engine with:
- Asynchronous event loop architecture
- Comprehensive health checks
- Graceful shutdown handling
- Market hours validation
- Connection monitoring
"""

import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from robo_trader.config import Config, TradingMode
from robo_trader.database import TradingDatabase
from robo_trader.execution import AbstractExecutor, PaperExecutor
from robo_trader.ibkr_client import IBKRClient
from robo_trader.logger import get_logger
from robo_trader.portfolio import Portfolio
from robo_trader.risk import RiskManager, create_risk_manager_from_config
from robo_trader.correlation import CorrelationTracker
from robo_trader.strategy_manager import StrategyManager, create_default_manager

logger = get_logger(__name__)


class EngineState(Enum):
    """Trading engine state."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck:
    """Health check result."""
    def __init__(self, name: str, status: HealthStatus, message: str = ""):
        self.name = name
        self.status = status
        self.message = message
        self.timestamp = datetime.now()


class TradingEngine:
    """
    Asynchronous trading engine with comprehensive monitoring.
    
    Features:
    - Event-driven architecture
    - Health monitoring
    - Graceful shutdown
    - Market hours aware
    - Automatic recovery
    """
    
    def __init__(self, config: Config):
        """
        Initialize trading engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.state = EngineState.INITIALIZING
        
        # Core components
        self.ibkr_client: Optional[IBKRClient] = None
        self.database: Optional[TradingDatabase] = None
        self.risk_manager: Optional[RiskManager] = None
        self.executor: Optional[AbstractExecutor] = None
        self.portfolio: Optional[Portfolio] = None
        self.strategy_manager: Optional[StrategyManager] = None
        self.correlation_tracker: Optional[CorrelationTracker] = None
        
        # Event loop and tasks
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.tasks: Dict[str, asyncio.Task] = {}
        self.shutdown_event = asyncio.Event()
        
        # Health monitoring
        self.health_checks: Dict[str, HealthCheck] = {}
        self.last_health_check = datetime.now()
        
        # Performance metrics
        self.metrics = {
            "trades_executed": 0,
            "orders_placed": 0,
            "errors_count": 0,
            "last_trade_time": None,
            "engine_start_time": None,
            "uptime_seconds": 0,
        }
        
        # Market hours (EST/EDT)
        self.market_open = time(9, 30)  # 9:30 AM
        self.market_close = time(16, 0)  # 4:00 PM
        self.premarket_open = time(4, 0)  # 4:00 AM
        self.aftermarket_close = time(20, 0)  # 8:00 PM
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            logger.info("Initializing trading engine...")
            self.state = EngineState.INITIALIZING
            self.metrics["engine_start_time"] = datetime.now()
            
            # Initialize database
            self.database = TradingDatabase()
            
            # Initialize IBKR client
            self.ibkr_client = IBKRClient(
                self.config.ibkr.host,
                self.config.ibkr.port,
                self.config.ibkr.client_id,
            )
            
            # Connect to IBKR
            await self._connect_ibkr()
            
            # Initialize risk manager
            self.risk_manager = create_risk_manager_from_config(self.config)
            
            # Initialize executor
            if self.config.execution.mode == TradingMode.PAPER:
                self.executor = PaperExecutor()
            else:
                # TODO: Implement live executor
                raise NotImplementedError("Live trading not yet implemented")
            
            # Initialize portfolio
            self.portfolio = Portfolio(self.config.default_cash)
            
            # Initialize strategy manager
            self.strategy_manager = create_default_manager()
            
            # Initialize correlation tracker
            self.correlation_tracker = CorrelationTracker(
                lookback_days=60,
                correlation_threshold=self.config.risk.correlation_limit,
            )
            
            # Run initial health check
            await self._run_health_checks()
            
            if self._is_healthy():
                self.state = EngineState.READY
                logger.info("Trading engine initialized successfully")
            else:
                self.state = EngineState.ERROR
                logger.error("Trading engine initialization failed health checks")
                
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            self.state = EngineState.ERROR
            raise
    
    async def _connect_ibkr(self, max_retries: int = 3) -> None:
        """Connect to IBKR with retry logic."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to IBKR (attempt {attempt + 1}/{max_retries})...")
                await self.ibkr_client.connect(
                    readonly=self.config.ibkr.readonly,
                    timeout=self.config.ibkr.timeout,
                )
                logger.info("Successfully connected to IBKR")
                return
            except Exception as e:
                logger.error(f"IBKR connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    async def start(self) -> None:
        """Start the trading engine."""
        if self.state != EngineState.READY:
            raise RuntimeError(f"Cannot start engine in state {self.state}")
        
        logger.info("Starting trading engine...")
        self.state = EngineState.RUNNING
        self.loop = asyncio.get_event_loop()
        
        # Start core tasks
        self.tasks["health_monitor"] = asyncio.create_task(self._health_monitor_loop())
        self.tasks["risk_monitor"] = asyncio.create_task(self._risk_monitor_loop())
        self.tasks["trading"] = asyncio.create_task(self._trading_loop())
        self.tasks["market_hours"] = asyncio.create_task(self._market_hours_loop())
        
        # Start data streaming if enabled
        if self.config.data.enable_real_time:
            self.tasks["data_stream"] = asyncio.create_task(self._data_streaming_loop())
        
        logger.info(f"Started {len(self.tasks)} async tasks")
        
        # Wait for shutdown signal
        await self.shutdown_event.wait()
    
    async def _health_monitor_loop(self) -> None:
        """Monitor system health continuously."""
        while self.state == EngineState.RUNNING:
            try:
                await self._run_health_checks()
                
                if not self._is_healthy():
                    logger.warning("System health degraded")
                    if self.health_checks.get("critical", HealthCheck("critical", HealthStatus.HEALTHY)).status == HealthStatus.UNHEALTHY:
                        logger.error("Critical health check failed, initiating shutdown")
                        await self.emergency_shutdown()
                        break
                
                await asyncio.sleep(self.config.monitoring.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.metrics["errors_count"] += 1
                await asyncio.sleep(10)
    
    async def _risk_monitor_loop(self) -> None:
        """Monitor risk metrics continuously."""
        while self.state == EngineState.RUNNING:
            try:
                # Check for emergency shutdown conditions
                if self.risk_manager and self.risk_manager.should_emergency_shutdown():
                    logger.error("Risk manager triggered emergency shutdown")
                    await self.emergency_shutdown()
                    break
                
                # Update portfolio risk metrics
                if self.portfolio and self.risk_manager:
                    positions = {
                        symbol: pos
                        for symbol, pos in self.portfolio.positions.items()
                    }
                    
                    if positions:
                        # Get current prices
                        current_prices = await self._get_current_prices(list(positions.keys()))
                        
                        # Calculate risk metrics
                        metrics = self.risk_manager.calculate_risk_metrics(
                            positions,
                            current_prices,
                        )
                        
                        # Log risk metrics
                        logger.debug(f"Portfolio heat: {metrics.portfolio_heat:.2%}")
                        logger.debug(f"Portfolio beta: {metrics.portfolio_beta:.2f}")
                        
                        # Check for violations
                        if metrics.portfolio_heat > self.config.risk.max_portfolio_heat:
                            logger.warning(f"Portfolio heat ({metrics.portfolio_heat:.2%}) exceeds limit")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Risk monitor error: {e}")
                self.metrics["errors_count"] += 1
                await asyncio.sleep(10)
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.state == EngineState.RUNNING:
            try:
                # Check if market is open
                if not self._is_market_open():
                    await asyncio.sleep(60)
                    continue
                
                # Process each symbol
                for symbol in self.config.symbols:
                    if self.state != EngineState.RUNNING:
                        break
                    
                    await self._process_symbol(symbol)
                
                # Update metrics
                self.metrics["uptime_seconds"] = (
                    datetime.now() - self.metrics["engine_start_time"]
                ).total_seconds()
                
                # Sleep before next iteration
                await asyncio.sleep(self.config.execution.order_timeout_seconds)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                self.metrics["errors_count"] += 1
                await asyncio.sleep(10)
    
    async def _market_hours_loop(self) -> None:
        """Monitor market hours and handle transitions."""
        while self.state == EngineState.RUNNING:
            try:
                now = datetime.now()
                current_time = now.time()
                
                # Check for market open
                if current_time >= self.market_open and current_time < self.market_close:
                    if not self._is_market_open_cached:
                        logger.info("Market opened")
                        self._is_market_open_cached = True
                        # Reset daily counters
                        if self.risk_manager:
                            self.risk_manager.reset_daily_counters()
                
                # Check for market close
                elif self._is_market_open_cached:
                    logger.info("Market closed")
                    self._is_market_open_cached = False
                    # End of day tasks
                    await self._end_of_day_tasks()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Market hours monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _data_streaming_loop(self) -> None:
        """Stream real-time market data."""
        while self.state == EngineState.RUNNING:
            try:
                # TODO: Implement real-time data streaming
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Data streaming error: {e}")
                self.metrics["errors_count"] += 1
                await asyncio.sleep(10)
    
    async def _process_symbol(self, symbol: str) -> None:
        """
        Process trading logic for a symbol.
        
        Args:
            symbol: Trading symbol
        """
        try:
            # Fetch market data
            bars = await self.ibkr_client.fetch_recent_bars(
                symbol,
                duration=self.config.data.bar_size,
                bar_size=self.config.data.bar_size,
            )
            
            if bars.empty:
                return
            
            # Store market data
            if self.database:
                # TODO: Store market data
                pass
            
            # Update correlation tracker
            if self.correlation_tracker and len(bars) > 0:
                self.correlation_tracker.add_price_series(
                    symbol,
                    bars["close"],
                )
            
            # Generate trading signals
            signal = self.strategy_manager.evaluate_symbol(symbol, bars)
            
            # Execute trades based on signals
            # TODO: Implement trade execution logic
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            self.metrics["errors_count"] += 1
    
    async def _get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for symbols."""
        prices = {}
        for symbol in symbols:
            try:
                # TODO: Implement real-time price fetching
                prices[symbol] = 0.0
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
        return prices
    
    async def _end_of_day_tasks(self) -> None:
        """Perform end of day tasks."""
        logger.info("Running end of day tasks...")
        
        try:
            # Export portfolio
            if self.portfolio:
                self.portfolio.export_csv(f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv")
            
            # Log daily metrics
            logger.info(f"Daily metrics: {self.metrics}")
            
            # Update correlation matrix
            if self.correlation_tracker:
                corr_summary = self.correlation_tracker.get_correlation_summary()
                logger.info(f"Correlation summary: {corr_summary}")
            
        except Exception as e:
            logger.error(f"End of day tasks error: {e}")
    
    async def _run_health_checks(self) -> None:
        """Run all health checks."""
        # Check IBKR connection
        if self.ibkr_client:
            if self.ibkr_client.ib.isConnected():
                self.health_checks["ibkr"] = HealthCheck(
                    "ibkr",
                    HealthStatus.HEALTHY,
                    "Connected to IBKR"
                )
            else:
                self.health_checks["ibkr"] = HealthCheck(
                    "ibkr",
                    HealthStatus.UNHEALTHY,
                    "Disconnected from IBKR"
                )
        
        # Check database
        if self.database:
            try:
                # TODO: Implement database health check
                self.health_checks["database"] = HealthCheck(
                    "database",
                    HealthStatus.HEALTHY,
                    "Database operational"
                )
            except Exception as e:
                self.health_checks["database"] = HealthCheck(
                    "database",
                    HealthStatus.UNHEALTHY,
                    str(e)
                )
        
        # Check risk manager
        if self.risk_manager:
            if self.risk_manager.emergency_shutdown_triggered:
                self.health_checks["risk"] = HealthCheck(
                    "risk",
                    HealthStatus.UNHEALTHY,
                    "Emergency shutdown triggered"
                )
            else:
                violations = len(self.risk_manager.violations)
                if violations > 10:
                    self.health_checks["risk"] = HealthCheck(
                        "risk",
                        HealthStatus.DEGRADED,
                        f"{violations} risk violations"
                    )
                else:
                    self.health_checks["risk"] = HealthCheck(
                        "risk",
                        HealthStatus.HEALTHY,
                        "Risk limits normal"
                    )
        
        # Check system resources
        # TODO: Implement resource checks (memory, CPU, disk)
        
        self.last_health_check = datetime.now()
    
    def _is_healthy(self) -> bool:
        """Check if system is healthy."""
        for check in self.health_checks.values():
            if check.status == HealthStatus.UNHEALTHY:
                return False
        return True
    
    def _is_market_open(self) -> bool:
        """Check if market is open."""
        now = datetime.now()
        current_time = now.time()
        
        # Check if weekend
        if now.weekday() in [5, 6]:  # Saturday, Sunday
            return False
        
        # Check market hours
        if self.config.execution.mode == TradingMode.PAPER:
            # Paper trading can run extended hours
            return current_time >= self.premarket_open and current_time <= self.aftermarket_close
        else:
            # Live trading only during regular hours
            return current_time >= self.market_open and current_time <= self.market_close
    
    _is_market_open_cached = False
    
    async def pause(self) -> None:
        """Pause trading engine."""
        if self.state != EngineState.RUNNING:
            return
        
        logger.info("Pausing trading engine...")
        self.state = EngineState.PAUSED
    
    async def resume(self) -> None:
        """Resume trading engine."""
        if self.state != EngineState.PAUSED:
            return
        
        logger.info("Resuming trading engine...")
        self.state = EngineState.RUNNING
    
    async def shutdown(self, timeout: int = 30) -> None:
        """
        Gracefully shutdown the trading engine.
        
        Args:
            timeout: Shutdown timeout in seconds
        """
        if self.state == EngineState.STOPPED:
            return
        
        logger.info("Initiating graceful shutdown...")
        self.state = EngineState.STOPPING
        
        # Cancel all tasks
        for name, task in self.tasks.items():
            if not task.done():
                logger.debug(f"Cancelling task: {name}")
                task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        # Close connections
        if self.ibkr_client:
            self.ibkr_client.disconnect()
        
        # Save final state
        if self.portfolio:
            self.portfolio.export_csv("portfolio_final.csv")
        
        self.state = EngineState.STOPPED
        self.shutdown_event.set()
        logger.info("Trading engine shutdown complete")
    
    async def emergency_shutdown(self) -> None:
        """Emergency shutdown - immediate stop."""
        logger.error("EMERGENCY SHUTDOWN INITIATED")
        
        # Flatten all positions if configured
        if self.config.execution.mode == TradingMode.LIVE:
            logger.warning("Would flatten all positions in live mode")
            # TODO: Implement position flattening
        
        # Force shutdown
        self.state = EngineState.STOPPED
        self.shutdown_event.set()
        
        # Cancel all tasks immediately
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        
        # Disconnect immediately
        if self.ibkr_client:
            self.ibkr_client.disconnect()
        
        logger.error("Emergency shutdown complete")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "state": self.state.value,
            "health_checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                }
                for name, check in self.health_checks.items()
            },
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "is_healthy": self._is_healthy(),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        return {
            **self.metrics,
            "state": self.state.value,
            "tasks_running": len([t for t in self.tasks.values() if not t.done()]),
            "health_status": self._is_healthy(),
        }


async def create_and_start_engine(config: Config) -> TradingEngine:
    """
    Create and start a trading engine.
    
    Args:
        config: Configuration object
        
    Returns:
        Running TradingEngine instance
    """
    engine = TradingEngine(config)
    await engine.initialize()
    
    if engine.state == EngineState.READY:
        await engine.start()
    else:
        raise RuntimeError("Engine failed to initialize")
    
    return engine