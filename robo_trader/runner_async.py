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
import os
import subprocess
import sys

# Load .env EARLY before any os.getenv() calls at module level
from dotenv import load_dotenv  # isort:skip

load_dotenv()  # noqa: E402 - must run before imports that use os.getenv()
from dataclasses import dataclass  # isort:skip  # noqa: E402
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd
from ib_async import Stock

from .analysis.correlation_integration import AsyncCorrelationManager, CorrelationBasedPositionSizer
from .config import load_config
from .correlation import CorrelationTracker
from .database_async import AsyncTradingDatabase
from .execution import Order, PaperExecutor
from .logger import get_logger
from .market_hours import (
    get_market_session,
    is_extended_hours,
    is_market_open,
    seconds_until_market_open,
)
from .monitoring.performance import PerformanceMonitor, Timer
from .monitoring.production_monitor import ProductionMonitor
from .utils.robust_connection import CircuitBreakerConfig, connect_ibkr_robust

# Import WebSocket client for real-time updates
try:
    from .websocket_client import ws_client

    WEBSOCKET_ENABLED = True
except ImportError:
    ws_client = None
    WEBSOCKET_ENABLED = False
from .circuit_breaker import CircuitBreaker
from .exceptions import KillSwitchTriggeredError
from .portfolio import Portfolio, PositionSnapshot  # Import Portfolio class from portfolio.py file
from .portfolio_pkg.portfolio_manager import AllocationMethod, MultiStrategyPortfolioManager
from .risk.advanced_risk import AdvancedRiskManager, risk_monitor_task
from .risk_manager import Position, RiskManager
from .stop_loss_monitor import StopLossMonitor, StopType
from .strategies import MLStrategy, sma_crossover_signals
from .strategies.ml_enhanced_strategy import MLEnhancedStrategy
from .utils.connection_recovery import OrderRateLimiter

# Initialize logger early for use in import error handling
logger = get_logger(__name__)

# Import AI analyst for news-driven trading
try:
    from .ai_analyst import AIAnalyst, MarketSentiment, create_analyst
    from .news_fetcher import fetch_rss_news

    AI_ANALYST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI analyst not available: {e}")
    AI_ANALYST_AVAILABLE = False
    create_analyst = None
from .utils.pricing import PrecisePricing  # noqa: E402
from .utils.secure_config import SecureConfig  # noqa: E402

# Import mean reversion strategies if available

try:
    from .strategies.mean_reversion import MeanReversionStrategy
    from .strategies.pairs_trading import CointegrationPairsStrategy, StatisticalArbitrageStrategy

    MEAN_REVERSION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Mean reversion strategies not available: {e}")
    MEAN_REVERSION_AVAILABLE = False


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


# Check if extended hours trading is enabled
ENABLE_EXTENDED_HOURS = os.getenv("ENABLE_EXTENDED_HOURS", "false").lower() in ("true", "1", "yes")


def is_trading_allowed() -> bool:
    """Check if trading is currently allowed (regular hours or extended hours if enabled)."""
    return is_market_open() or (ENABLE_EXTENDED_HOURS and is_extended_hours())


class AsyncRunner:
    """Enhanced async runner with parallel processing capabilities."""

    def __init__(
        self,
        duration: str = "1 D",  # Reduced from 10D for faster cycles
        bar_size: str = "1 min",  # 1-minute bars for near real-time
        sma_fast: int = 10,
        sma_slow: int = 20,
        slippage_bps: float = 0.0,
        max_order_notional: Optional[float] = None,
        max_daily_notional: Optional[float] = None,
        default_cash: Optional[float] = None,
        max_concurrent_symbols: int = 8,
        use_correlation_sizing: bool = True,
        max_correlation: float = 0.7,
        use_ml_strategy: bool = False,
        use_ml_enhanced: bool = None,  # Auto-detect if None
        use_smart_execution: bool = None,  # Auto-detect if None
        use_advanced_risk: bool = None,  # Auto-detect if None
        portfolio_id: str = "default",  # Multi-portfolio support
    ):
        self.portfolio_id = portfolio_id
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
        self.use_ml_strategy = use_ml_strategy
        # Auto-detect ML enhanced if not explicitly set
        if use_ml_enhanced is None:
            self.use_ml_enhanced = os.getenv("ML_ENHANCED_ENABLED", "true").lower() == "true"
        else:
            self.use_ml_enhanced = use_ml_enhanced
        # Auto-detect smart execution if not explicitly set
        if use_smart_execution is None:
            self.use_smart_execution = (
                os.getenv("SMART_EXECUTION_ENABLED", "true").lower() == "true"
            )
        else:
            self.use_smart_execution = use_smart_execution

        # Auto-detect advanced risk management if not explicitly set
        if use_advanced_risk is None:
            self.use_advanced_risk = os.getenv("ADVANCED_RISK_ENABLED", "true").lower() == "true"
        else:
            self.use_advanced_risk = use_advanced_risk

        # Will be initialized in setup
        self.cfg = None
        self.ib = None
        self.db = None
        self.risk = None
        self.advanced_risk = None  # Advanced risk manager with Kelly sizing
        self.risk_monitor_task = None  # Background risk monitoring task
        self.subprocess_monitor_task = None  # Background subprocess health monitoring task
        self.executor = None
        self.portfolio = None
        self.portfolio_manager: Optional[MultiStrategyPortfolioManager] = None
        self.positions: Dict[str, Position] = {}
        self.latest_prices: Dict[str, float] = {}
        self.daily_pnl = 0.0
        self.daily_executed_notional = 0.0
        self.session_start_equity: Optional[float] = None  # Captured after loading positions
        self.monitor = PerformanceMonitor()

        # Production monitoring (Phase 4 P2)
        self.production_monitor = None
        self.enable_production_monitoring = (
            os.getenv("PRODUCTION_MONITORING", "true").lower() == "true"
        )

        # Position locks to prevent race conditions
        self._position_locks: Dict[str, asyncio.Lock] = {}
        self._position_lock_manager = asyncio.Lock()

        # CRITICAL: Track pending orders to prevent duplicate buys during parallel processing
        # This set is checked before any BUY order is placed
        self._pending_orders: set = set()
        self._pending_orders_lock = asyncio.Lock()

        # ADDITIONAL PROTECTION: Track symbols that have executed BUY orders this cycle
        # This is a more aggressive duplicate prevention mechanism
        self._cycle_executed_buys: set = set()
        self._cycle_executed_buys_lock = asyncio.Lock()

        # Correlation components
        self.correlation_tracker = None
        self.position_sizer = None
        self.correlation_manager = None

        # ML Strategy
        self.ml_strategy = None

        # Stop-loss monitoring
        # NOTE: These are defaults; actual values loaded from self.cfg in setup()
        self.stop_loss_monitor = None
        self.enable_stop_loss = True  # Always enabled for safety
        self.stop_loss_percent = 0.02  # Default 2%, updated from config in setup()
        self.use_trailing_stop = True  # Default to trailing, updated from config in setup()
        self.trailing_stop_pct = 0.05  # Default 5%, updated from config in setup()
        self.ml_enhanced_strategy = None
        self.active_strategy_name: Optional[str] = None
        self._ml_predictions: Dict[str, Dict] = {}  # Track ML predictions for dashboard

        # Circuit breaker for order execution protection
        self.circuit_breaker = CircuitBreaker(
            name="order_execution",
            failure_threshold=5,  # Allow 5 failures before opening
            recovery_timeout=60,  # Wait 60 seconds before attempting recovery
            half_open_requests=3,  # Allow 3 test requests in half-open state
        )

        # Rate limiter to prevent hitting IB API limits
        self.rate_limiter = OrderRateLimiter(
            max_per_second=2, max_per_minute=50  # Conservative limit  # IB's typical limit
        )

        # Mean reversion strategies
        self.mean_reversion_strategy = None
        self.pairs_strategy = None
        self.stat_arb_strategy = None

        # AI Analyst for news-driven trading
        self.ai_analyst = None
        self.use_ai_trading = os.getenv("AI_TRADING_ENABLED", "true").lower() == "true"
        self.news_cache = {}  # Cache recent news to avoid re-fetching
        self.last_news_fetch = None
        self._ai_opportunities = {}  # AI-identified buying opportunities
        # Cache for pairs/stat arb analysis with size limit
        from collections import OrderedDict

        self.market_data_cache = OrderedDict()  # LRU-style cache
        self.max_cache_size = 100  # Maximum number of symbols to cache

        # Symbol to sector mapping (can be enhanced with external data source)
        self.symbol_sectors = {
            # Technology
            "AAPL": "Technology",
            "MSFT": "Technology",
            "GOOGL": "Technology",
            "META": "Technology",
            "NVDA": "Technology",
            "AMD": "Technology",
            "INTC": "Technology",
            "CSCO": "Technology",
            "ORCL": "Technology",
            "PLTR": "Technology",
            "UPST": "Technology",
            "SOFI": "Technology",
            # Financials
            "JPM": "Financials",
            "BAC": "Financials",
            "WFC": "Financials",
            "GS": "Financials",
            "MS": "Financials",
            "C": "Financials",
            "AXP": "Financials",
            "V": "Financials",
            "MA": "Financials",
            # Healthcare
            "JNJ": "Healthcare",
            "PFE": "Healthcare",
            "UNH": "Healthcare",
            "CVS": "Healthcare",
            "ABBV": "Healthcare",
            "MRK": "Healthcare",
            "TMO": "Healthcare",
            "ABT": "Healthcare",
            "LLY": "Healthcare",
            # Energy
            "XOM": "Energy",
            "CVX": "Energy",
            "COP": "Energy",
            "SLB": "Energy",
            "EOG": "Energy",
            "PXD": "Energy",
            "CEG": "Energy",
            "VRT": "Energy",
            # Consumer
            "AMZN": "Consumer",
            "TSLA": "Consumer",
            "WMT": "Consumer",
            "HD": "Consumer",
            "NKE": "Consumer",
            "MCD": "Consumer",
            "SBUX": "Consumer",
            "TGT": "Consumer",
            "COST": "Consumer",
            # Industrials
            "BA": "Industrials",
            "CAT": "Industrials",
            "GE": "Industrials",
            "LMT": "Industrials",
            "UPS": "Industrials",
            "RTX": "Industrials",
            "HON": "Industrials",
            "DE": "Industrials",
            "MMM": "Industrials",
            # Communications
            "DIS": "Communications",
            "NFLX": "Communications",
            "CMCSA": "Communications",
            "T": "Communications",
            "VZ": "Communications",
            "TMUS": "Communications",
            # Materials
            "LIN": "Materials",
            "APD": "Materials",
            "SHW": "Materials",
            "ECL": "Materials",
            "DD": "Materials",
            "NEM": "Materials",
            # Real Estate
            "PLD": "Real Estate",
            "AMT": "Real Estate",
            "CCI": "Real Estate",
            "EQIX": "Real Estate",
            "PSA": "Real Estate",
            "SPG": "Real Estate",
            "OPEN": "Real Estate",
            # Utilities
            "NEE": "Utilities",
            "SO": "Utilities",
            "DUK": "Utilities",
            "D": "Utilities",
            "AEP": "Utilities",
            "EXC": "Utilities",
            # Crypto/Blockchain
            "CORZ": "Crypto",
            "WULF": "Crypto",
            "RIOT": "Crypto",
            "MARA": "Crypto",
            "HUT": "Crypto",
            "BTBT": "Crypto",
            # AI/ML Companies
            "IXHL": "AI/Tech",
            "NUAI": "AI/Tech",
            "BZAI": "AI/Tech",
            "ELTP": "AI/Tech",
            "SDGR": "AI/Tech",
            "APLD": "AI/Tech",
            # Biotech
            "HTFL": "Biotech",
            "TEM": "Biotech",
            # ETFs
            "SPY": "ETF",
            "QQQ": "ETF",
            "IWM": "ETF",
            "DIA": "ETF",
            "VTI": "ETF",
            "VOO": "ETF",
        }

    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector classification for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Sector name or 'Unknown' if not found
        """
        return self.symbol_sectors.get(symbol, "Unknown")

    async def _get_position_lock(self, symbol: str) -> asyncio.Lock:
        """Get or create a lock for a specific symbol."""
        async with self._position_lock_manager:
            if symbol not in self._position_locks:
                self._position_locks[symbol] = asyncio.Lock()
            return self._position_locks[symbol]

    async def _place_order_with_circuit_breaker(self, order: Order):
        """
        Execute order with circuit breaker protection.

        Checks circuit breaker state before execution and records
        success/failure for fault tolerance monitoring.
        """
        # Check if circuit breaker allows the request
        if not await self.circuit_breaker.can_proceed():
            logger.error(f"Circuit breaker OPEN - order rejected for {order.symbol}")
            return SimpleNamespace(ok=False, message="Circuit breaker open", fill_price=None)

        # Apply rate limiting before order execution
        await self.rate_limiter.acquire()
        logger.debug(f"Rate limit acquired for {order.symbol} order")

        try:
            # Execute the order
            with Timer("order_execution", self.monitor):
                result = self.executor.place_order(order)

            # Record success/failure with circuit breaker
            if result.ok:
                await self.circuit_breaker.record_success()
                logger.debug(
                    f"Order successful for {order.symbol} - circuit breaker recorded success"
                )
            else:
                await self.circuit_breaker.record_failure()
                logger.warning(
                    f"Order failed for {order.symbol} - circuit breaker recorded failure"
                )

            return result

        except Exception as e:
            # Record failure and re-raise
            await self.circuit_breaker.record_failure(e)
            logger.error(
                f"Order execution exception for {order.symbol} - circuit breaker recorded failure: {e}"
            )
            raise

    async def _update_position_atomic(
        self, symbol: str, quantity: int, price: float, side: str
    ) -> bool:
        """Atomically update position with lock to prevent race conditions."""
        lock = await self._get_position_lock(symbol)
        async with lock:
            try:
                if side.upper() in ["BUY", "BUY_TO_COVER"]:
                    if symbol in self.positions:
                        # Update existing position
                        pos = self.positions[symbol]
                        if side.upper() == "BUY_TO_COVER" and pos.quantity < 0:
                            # Covering short position
                            new_qty = pos.quantity + quantity
                            if new_qty == 0:
                                del self.positions[symbol]
                            else:
                                self.positions[symbol] = Position(
                                    symbol, new_qty, float(pos.avg_price)
                                )
                        elif side.upper() == "BUY" and pos.quantity >= 0:
                            # Adding to long position
                            total_qty = pos.quantity + quantity
                            new_avg = (
                                float(pos.avg_price) * pos.quantity + price * quantity
                            ) / total_qty
                            self.positions[symbol] = Position(symbol, total_qty, new_avg)
                        else:
                            logger.error(
                                f"Invalid position update: {side} on {pos.quantity} shares of {symbol}"
                            )
                            return False
                    else:
                        # New long position
                        self.positions[symbol] = Position(symbol, quantity, price)

                elif side.upper() in ["SELL", "SELL_SHORT"]:
                    if symbol in self.positions:
                        pos = self.positions[symbol]
                        if side.upper() == "SELL" and pos.quantity > 0:
                            # Selling long position
                            if quantity >= pos.quantity:
                                del self.positions[symbol]
                            else:
                                remaining = pos.quantity - quantity
                                self.positions[symbol] = Position(
                                    symbol, remaining, float(pos.avg_price)
                                )
                        else:
                            logger.error(
                                f"Invalid position update: {side} on {pos.quantity} shares of {symbol}"
                            )
                            return False
                    elif side.upper() == "SELL_SHORT":
                        # New short position (negative quantity)
                        self.positions[symbol] = Position(symbol, -quantity, price)
                    else:
                        logger.error(f"Cannot {side} {symbol}: no existing position")
                        return False

                # Update portfolio (thread-safe async call)
                await self.portfolio.update_fill(symbol, side, quantity, price)

                # Update advanced risk manager if enabled
                if self.use_advanced_risk and self.advanced_risk:
                    self.advanced_risk.update_position(symbol, quantity, price, side)

                return True

            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
                return False

    async def setup(self):
        """Initialize all components."""
        # Skip setup if already connected (for persistent connections)
        if hasattr(self, "_setup_complete") and self._setup_complete:
            # Just verify connection is still alive
            if hasattr(self, "ib") and self.ib:
                try:
                    if hasattr(self.ib, "ping"):
                        ping_ok = await self.ib.ping()
                        if ping_ok:
                            logger.info("✓ Persistent IBKR connection still active")
                            return
                    elif hasattr(self.ib, "isConnected") and self.ib.isConnected():
                        logger.info("✓ Persistent IBKR connection still active")
                        return
                except Exception as e:
                    logger.warning(f"Connection check failed: {e}, will reconnect")
                    self._setup_complete = False

        self.cfg = load_config()

        # Load stop-loss settings from validated config
        self.stop_loss_percent = self.cfg.risk.stop_loss_pct
        self.use_trailing_stop = self.cfg.risk.use_trailing_stop
        self.trailing_stop_pct = self.cfg.risk.trailing_stop_pct
        logger.info(
            f"Stop-loss config loaded: trailing={self.use_trailing_stop}, "
            f"trailing_pct={self.trailing_stop_pct:.1%}, fixed_pct={self.stop_loss_percent:.1%}"
        )

        # CRITICAL PRE-FLIGHT CHECK: Test IBKR connection before proceeding
        # This prevents the system from starting without a working connection
        logger.info("=" * 60)
        logger.info("PERFORMING IBKR PRE-FLIGHT CONNECTION CHECK")
        logger.info("=" * 60)

        # Test if configured TWS/IB Gateway port is open using lsof
        # IMPORTANT: Do NOT use socket.connect_ex() - it creates zombie connections
        # that block subsequent IBKR API handshakes!

        def test_port_open_lsof(port=7497):
            """Check if port is listening using lsof (no zombies)."""
            try:
                result = subprocess.run(
                    ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                return result.returncode == 0 and "LISTEN" in result.stdout
            except Exception as e:
                logger.error(f"Port test failed: {e}")
                return False

        host = self.cfg.ibkr.host
        port = self.cfg.ibkr.port

        if not test_port_open_lsof(port=port):
            logger.error(f"❌ IBKR PRE-FLIGHT CHECK FAILED")
            logger.error(f"Port {port} is not listening - TWS/IB Gateway not running")
            logger.error("Please ensure TWS or IB Gateway is running and configured properly:")
            logger.error("1. Start TWS/IB Gateway")
            logger.error("2. Enable API connections in Global Configuration")
            logger.error("3. Set correct port (7497 for paper, 7496 for live)")
            logger.error("4. Allow connections from localhost")
            logger.error("REFUSING TO START WITHOUT IBKR CONNECTION")
            sys.exit(1)

        logger.info(f"✓ Port {port} is open - proceeding to IBKR connect")

        # CRITICAL: Kill ALL zombie connections before attempting to connect
        # This is essential because Gateway-owned zombies block new API handshakes
        logger.info("=" * 60)
        logger.info("CLEANING UP ZOMBIE CONNECTIONS BEFORE CONNECT")
        logger.info("=" * 60)

        from .utils.robust_connection import (
            check_tws_zombie_connections,
            kill_tws_zombie_connections,
            restart_gateway_for_zombies,
        )

        # Check for zombies
        zombie_count, error_msg = check_tws_zombie_connections(port)
        if zombie_count > 0:
            logger.warning(f"Found {zombie_count} zombie connection(s) - cleaning up...")
            success, msg = kill_tws_zombie_connections(port)
            if success:
                logger.info(f"✓ {msg}")
            else:
                logger.warning(f"⚠️ {msg}")
                # Gateway-owned zombies require Gateway restart
                # NOTE: We do NOT auto-restart Gateway here because:
                # 1. Gateway restart requires 2FA which can't be completed in a subprocess
                # 2. The startup script (START_TRADER.sh) handles Gateway restarts properly
                # 3. If we get here, something created a zombie after the startup script checked
                if "Gateway zombies remain" in msg or "Gateway-owned" in msg:
                    logger.error("❌ Gateway-owned zombies detected - these block API handshakes")
                    logger.error(
                        "The dashboard or another process may have created a zombie connection."
                    )
                    logger.error("Please restart Gateway manually and try again:")
                    logger.error("  1. Kill this process: Ctrl+C")
                    logger.error(
                        "  2. Restart Gateway: python3 scripts/gateway_manager.py restart --paper"
                    )
                    logger.error("  3. Complete 2FA on your phone")
                    logger.error("  4. Run: ./START_TRADER.sh")
                    raise RuntimeError(
                        "Gateway-owned zombie connections detected. Manual Gateway restart required."
                    )
        else:
            logger.info("✓ No zombie connections found")

        # Log connection details with secure masking
        client_id_masked = SecureConfig.mask_value(self.cfg.ibkr.client_id)
        logger.info(
            f"Initializing robust connection to {host}:{port} with client_id: {client_id_masked}"
        )

        # Try to connect using robust connection with circuit breaker
        try:
            # Configure circuit breaker from env
            circuit_config = CircuitBreakerConfig(
                failure_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
                recovery_timeout=float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300")),
                success_threshold=2,
            )

            self.ib = await connect_ibkr_robust(
                host=host,
                port=port,
                client_id=self.cfg.ibkr.client_id,
                readonly=self.cfg.ibkr.readonly,
                timeout=self.cfg.ibkr.timeout,
                max_retries=2,  # Reduced from 5 to prevent zombie connection accumulation
                circuit_breaker_config=circuit_config,
                ssl_mode=self.cfg.ibkr.ssl_mode,
            )
            logger.info("✓ IBKR connection established successfully with robust connection")
            logger.info("=" * 60)

            # Skip subprocess health monitoring - we disconnect between cycles for stability
            # This avoids the health check triggering restarts and lock file contention
            logger.info(
                "✓ Using per-cycle connection mode (disconnect between cycles for stability)"
            )

        except ConnectionError as e:
            logger.error(f"❌ IBKR CONNECTION FAILED: {e}")
            logger.error("Cannot proceed without IBKR connection")
            sys.exit(1)

        # Initialize async database with context manager
        self._raw_db = AsyncTradingDatabase()
        await self._raw_db.initialize()

        # Wrap DB with portfolio-scoped proxy if using non-default portfolio
        from .multiuser.db_proxy import PortfolioScopedDB

        self.db = PortfolioScopedDB(self._raw_db, portfolio_id=self.portfolio_id)
        logger.info(f"Database initialized for portfolio: {self.portfolio_id}")

        # Initialize risk manager
        self.risk = RiskManager(
            max_daily_loss=self.cfg.risk.max_daily_loss_pct,
            max_position_risk_pct=self.cfg.risk.max_position_pct,
            max_symbol_exposure_pct=self.cfg.risk.max_sector_exposure_pct,
            max_leverage=self.cfg.risk.max_leverage,
            max_order_notional=self.max_order_notional or self.cfg.risk.max_order_notional,
            max_daily_notional=self.max_daily_notional or self.cfg.risk.max_daily_notional,
            max_open_positions=self.cfg.risk.max_open_positions,
        )

        # Initialize advanced risk manager with Kelly sizing and kill switches
        if self.use_advanced_risk:
            logger.info("Initializing advanced risk management with Kelly sizing and kill switches")
            self.advanced_risk = AdvancedRiskManager(
                config={
                    "starting_capital": (
                        self.default_cash
                        if self.default_cash is not None
                        else self.cfg.default_cash
                    ),
                    "max_position_pct": 0.1,
                    "max_risk_per_trade": 0.02,
                },
                enable_kelly=True,
                enable_correlation_limits=True,
                enable_kill_switch=True,
            )
            # Start background risk monitoring task
            self.risk_monitor_task = asyncio.create_task(
                risk_monitor_task(self.advanced_risk, interval=60)
            )
            logger.info(
                "Advanced risk management initialized with Kelly criterion and kill switches"
            )

        # Initialize executor with smart execution support
        smart_executor = None

        # Auto-enable smart execution for large orders or if explicitly enabled
        if self.use_smart_execution:
            from .smart_execution.smart_executor import SmartExecutor

            # Pass IBKR instance for real market data
            smart_executor = SmartExecutor(self.cfg, ibkr_client=self.ib)
            logger.info("Smart execution enabled with TWAP/VWAP/Iceberg algorithms")

        self.executor = PaperExecutor(
            slippage_bps=self.slippage_bps,
            smart_executor=smart_executor,
            use_smart_execution=self.use_smart_execution,
        )

        # Initialize stop-loss monitor (CRITICAL SAFETY COMPONENT)
        if self.enable_stop_loss:

            async def emergency_shutdown_callback(reason: str):
                """Emergency shutdown on stop-loss failure."""
                logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
                self.running = False
                if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                    self.advanced_risk.kill_switch.trigger(reason)
                # Cancel all active orders
                await self.cancel_all_orders()

            self.stop_loss_monitor = StopLossMonitor(
                executor=self.executor,
                risk_manager=self.risk,
                emergency_shutdown_callback=emergency_shutdown_callback,
                portfolio_id=self.portfolio_id,
            )
            await self.stop_loss_monitor.start_monitoring()
            if self.use_trailing_stop:
                logger.info(
                    f"TRAILING STOP enabled at {self.trailing_stop_pct:.1%} - "
                    f"stops follow price up, lock in profits!"
                )
            else:
                logger.info(f"Fixed stop-loss enabled at {self.stop_loss_percent:.1%}")

        # Initialize portfolio
        starting_cash = (
            self.default_cash if self.default_cash is not None else self.cfg.default_cash
        )
        self.portfolio = Portfolio(starting_cash)

        # Initialize Multi-Strategy Portfolio Manager (S3)
        try:
            self.portfolio_manager = MultiStrategyPortfolioManager(
                config=self.cfg,
                risk_manager=self.risk,
                allocation_method=AllocationMethod.ADAPTIVE,
                rebalance_frequency="daily",
                max_strategy_weight=0.4,
                min_strategy_weight=0.1,
            )
            self.portfolio_manager.update_capital(starting_cash)
        except Exception as e:
            logger.warning(f"Portfolio manager initialization failed: {e}")
            self.portfolio_manager = None

        # Initialize correlation components if enabled
        if self.use_correlation_sizing:
            self.correlation_tracker = CorrelationTracker(
                lookback_days=60, correlation_threshold=self.max_correlation
            )
            self.position_sizer = CorrelationBasedPositionSizer(
                correlation_tracker=self.correlation_tracker,
                max_correlation=self.max_correlation,
                correlation_penalty_factor=0.5,
                max_correlated_exposure=0.3,
            )
            self.correlation_manager = AsyncCorrelationManager(
                correlation_tracker=self.correlation_tracker, position_sizer=self.position_sizer
            )
            await self.correlation_manager.start(update_interval=300)
            logger.info("Correlation-based position sizing enabled")

        # Initialize ML Enhanced strategy if enabled or models exist
        if self.use_ml_enhanced or "ml_enhanced" in self.cfg.strategy.enabled_strategies:
            # Check if ML models exist
            models_dir = Path("models")
            has_models = models_dir.exists() and any(models_dir.glob("*.pkl"))

            if has_models or self.use_ml_enhanced:
                logger.info("Initializing ML Enhanced strategy...")
                # ML Enhanced uses the config directly
                self.ml_enhanced_strategy = MLEnhancedStrategy(self.cfg)
                # Initialize with historical data
                await self.ml_enhanced_strategy.initialize()
                logger.info("ML Enhanced strategy initialized successfully")
                self.active_strategy_name = "ML_Enhanced"
            else:
                logger.info(
                    "ML Enhanced enabled but no models found - will use fallback strategies"
                )
            if self.portfolio_manager:
                # Register ML Enhanced and a baseline SMA strategy for diversification
                self.portfolio_manager.register_strategy(
                    self.ml_enhanced_strategy, initial_weight=0.6
                )
                self.portfolio_manager.register_strategy(
                    SimpleNamespace(name="Baseline_SMA"), initial_weight=0.4
                )
        # Initialize regular ML strategy if enabled (and not using enhanced)
        elif self.use_ml_strategy:
            logger.info("Initializing ML strategy...")

            from .features.feature_pipeline import FeaturePipeline
            from .ml.model_selector import ModelSelector
            from .ml.model_trainer import ModelTrainer

            # Initialize feature pipeline
            feature_pipeline = FeaturePipeline(self.cfg)

            # Initialize model trainer
            model_trainer = ModelTrainer(config=self.cfg, model_dir=Path("trained_models"))

            # Initialize model selector with trained models
            model_selector = ModelSelector(
                model_trainer=model_trainer, model_dir=Path("trained_models")
            )

            # Create ML strategy with proper dependencies
            self.ml_strategy = MLStrategy(
                model_selector=model_selector,
                feature_pipeline=feature_pipeline,
                confidence_threshold=0.65,
                ensemble_agreement=0.6,
                use_regime_filter=True,
                position_size_method="kelly",
                max_position_pct=0.1,
                risk_per_trade=0.02,
                symbols=[],  # Will be set per symbol
                name="ML_Strategy",
            )

            # Initialize the strategy with empty historical data (models already trained)
            await self.ml_strategy.initialize({})
            logger.info("ML strategy initialized successfully")
            self.active_strategy_name = "ML_Strategy"
            if self.portfolio_manager:
                self.portfolio_manager.register_strategy(self.ml_strategy, initial_weight=0.6)
                self.portfolio_manager.register_strategy(
                    SimpleNamespace(name="Baseline_SMA"), initial_weight=0.4
                )
        else:
            # Fallback baseline
            self.active_strategy_name = "Baseline_SMA"
            if self.portfolio_manager:
                self.portfolio_manager.register_strategy(
                    SimpleNamespace(name="Baseline_SMA"), initial_weight=1.0
                )

        # Initialize mean reversion strategies if available and enabled
        if MEAN_REVERSION_AVAILABLE and "mean_reversion" in self.cfg.strategy.enabled_strategies:
            logger.info("Initializing mean reversion strategies...")
            try:
                # Initialize mean reversion strategy
                self.mean_reversion_strategy = MeanReversionStrategy(
                    symbols=self.cfg.symbols,
                    use_ml_enhancement=True,
                )

                # Initialize pairs trading if we have enough symbols
                if len(self.cfg.symbols) >= 2:
                    self.pairs_strategy = CointegrationPairsStrategy(
                        lookback_days=60,
                        use_ml_enhancement=True,
                    )
                    logger.info("Pairs trading strategy initialized")

                # Initialize statistical arbitrage
                if len(self.cfg.symbols) >= 5:
                    self.stat_arb_strategy = StatisticalArbitrageStrategy(
                        universe_size=min(len(self.cfg.symbols), 20),
                        max_positions=min(len(self.cfg.symbols) // 2, 10),
                        use_ml_ranking=True,
                    )
                    logger.info("Statistical arbitrage strategy initialized")

                logger.info("Mean reversion strategies initialized successfully")

                # Register with portfolio manager if available
                if self.portfolio_manager and self.mean_reversion_strategy:
                    self.portfolio_manager.register_strategy(
                        self.mean_reversion_strategy, initial_weight=0.2
                    )

            except Exception as e:
                logger.error(f"Failed to initialize mean reversion strategies: {e}")

        # Initialize AI Analyst for news-driven trading
        if self.use_ai_trading and AI_ANALYST_AVAILABLE and create_analyst:
            try:
                self.ai_analyst = create_analyst()
                if self.ai_analyst:
                    logger.info("AI Analyst initialized for news-driven trading")
                else:
                    logger.warning("AI Analyst not available (no API keys)")
            except Exception as e:
                logger.error(f"Failed to initialize AI analyst: {e}")
                self.ai_analyst = None

        # Initialize Production Monitoring (Phase 4 P2)
        if self.enable_production_monitoring:
            import json

            # Note: Path is already imported at module level (line 19)
            # Load monitoring config
            config_path = Path("config/monitoring_config.json")
            monitoring_config = {}
            if config_path.exists():
                with open(config_path) as f:
                    monitoring_config = json.load(f)

            # Initialize ProductionMonitor
            self.production_monitor = ProductionMonitor(
                config=monitoring_config.get("monitoring", {}),
                log_dir=Path(
                    monitoring_config.get("monitoring", {}).get("log_dir", "logs/monitoring")
                ),
                enable_alerts=monitoring_config.get("monitoring", {}).get("enable_alerts", True),
                enable_health_checks=monitoring_config.get("monitoring", {}).get(
                    "enable_health_checks", True
                ),
            )

            # Load alert configurations if available
            if "alerts" in monitoring_config:
                from .monitoring.production_monitor import Alert, AlertSeverity, MetricType

                for alert_name, alert_config in monitoring_config["alerts"].items():
                    alert = Alert(
                        name=alert_name,
                        metric_type=MetricType(alert_config["metric_type"]),
                        threshold=alert_config["threshold"],
                        comparison=alert_config["comparison"],
                        severity=AlertSeverity(alert_config["severity"]),
                        cooldown_minutes=alert_config.get("cooldown_minutes", 5),
                        message_template=alert_config.get("message_template", ""),
                        active=alert_config.get("active", True),
                    )
                    self.production_monitor.alert_manager.add_alert(alert)

            # Start monitoring
            await self.production_monitor.start(
                interval=monitoring_config.get("monitoring", {}).get("interval", 60)
            )
            logger.info("Production monitoring started with health checks and alerts")

        # Load existing positions from database to prevent duplicate buying
        await self.load_existing_positions()

        # Mark setup as complete for persistent connections
        self._setup_complete = True
        logger.info("AsyncRunner setup complete")

    async def load_existing_positions(self):
        """Load existing positions and account state from database on startup."""
        from decimal import Decimal

        try:
            # Load account state (cash, realized_pnl) from database
            account_info = await self.db.get_account_info()
            if account_info:
                db_cash = account_info.get("cash")
                db_realized_pnl = account_info.get("realized_pnl", 0.0)
                if db_cash is not None:
                    self.portfolio.cash = Decimal(str(db_cash))
                    self.portfolio.realized_pnl = Decimal(str(db_realized_pnl or 0.0))
                    logger.info(
                        f"Loaded account state: cash=${db_cash:,.2f}, "
                        f"realized_pnl=${db_realized_pnl or 0:,.2f}"
                    )

            # Load positions from database
            positions_data = await self.db.get_positions()
            for pos in positions_data:
                quantity = pos.get("quantity", 0)
                if quantity == 0:
                    continue

                symbol = pos["symbol"]
                avg_cost = pos.get("avg_cost", pos.get("price", 0))

                # Create Position object and add to runner's positions dict
                self.positions[symbol] = Position(symbol, quantity, avg_cost)

                # CRITICAL: Also sync to Portfolio object for equity calculation
                # Portfolio uses PositionSnapshot with Decimal avg_price
                avg_price_decimal = Decimal(str(avg_cost))
                self.portfolio.positions[symbol] = PositionSnapshot(
                    symbol, quantity, avg_price_decimal
                )

                logger.info(
                    f"Loaded existing position: {symbol} qty={quantity} avg_cost=${avg_cost:.2f}"
                )

            # CRITICAL FIX: Create stop-loss orders for existing positions
            # Without this, positions opened in previous sessions have NO stop-loss protection!
            if self.positions and self.stop_loss_monitor:
                logger.info(
                    f"Creating stop-loss orders for {len(self.positions)} existing positions..."
                )
                for symbol, pos in self.positions.items():
                    try:
                        # Create a position with float avg_price for stop-loss monitor
                        # (stop_loss_monitor uses float math internally)
                        float_pos = Position(
                            symbol=pos.symbol,
                            quantity=pos.quantity,
                            avg_price=float(pos.avg_price),
                            entry_time=pos.entry_time,
                        )
                        entry_price = float(pos.avg_price)
                        if self.use_trailing_stop:
                            await self.stop_loss_monitor.add_stop_loss(
                                symbol=symbol,
                                position=float_pos,
                                stop_percent=self.trailing_stop_pct,
                                stop_type=StopType.TRAILING_PERCENT,
                                trailing_percent=self.trailing_stop_pct,
                            )
                            stop_price = entry_price * (1 - self.trailing_stop_pct)
                            logger.info(
                                f"Trailing stop created for existing position {symbol} at {self.trailing_stop_pct:.1%} "
                                f"(entry=${entry_price:.2f}, initial stop=${stop_price:.2f})"
                            )
                        else:
                            await self.stop_loss_monitor.add_stop_loss(
                                symbol=symbol,
                                position=float_pos,
                                stop_percent=self.stop_loss_percent,
                                stop_type=StopType.FIXED,
                            )
                            stop_price = entry_price * (1 - self.stop_loss_percent)
                            logger.info(
                                f"Fixed stop-loss created for existing position {symbol} at {self.stop_loss_percent:.1%} "
                                f"(entry=${entry_price:.2f}, stop=${stop_price:.2f})"
                            )
                    except Exception as e:
                        logger.error(
                            f"Failed to create stop-loss for existing position {symbol}: {e}"
                        )

            if self.positions:
                logger.info(
                    f"Loaded {len(self.positions)} existing positions from database: {list(self.positions.keys())}"
                )
            else:
                logger.info("No existing positions found in database")

            # Capture session start equity for kill switch (after positions loaded)
            # Build market prices from loaded positions (use avg_price as proxy at startup)
            market_prices = {sym: float(pos.avg_price) for sym, pos in self.positions.items()}
            self.session_start_equity = float(await self.portfolio.equity(market_prices))
            logger.info(f"Session start equity: ${self.session_start_equity:,.2f}")

        except Exception as e:
            logger.error(f"Failed to load existing positions from database: {e}")
            logger.warning("Starting with empty positions - may result in duplicate trades!")
            # Fallback to config default if load fails
            self.session_start_equity = float(self.cfg.default_cash)

    async def _fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical bars from IBKR and return normalized DataFrame.

        Args:
            symbol: Ticker symbol
            duration: IB duration string
            bar_size: IB bar size
            what_to_show: Data type to request
            use_rth: Restrict to regular trading hours

        Returns:
            DataFrame with OHLCV columns
        """
        if not self.ib:
            raise ConnectionError("Not connected to IBKR")

        # Check if subprocess client or legacy IB client
        if hasattr(self.ib, "get_historical_bars"):
            # Subprocess client - use async method
            bars = await self.ib.get_historical_bars(
                symbol=symbol,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

            if not bars:
                return pd.DataFrame()

            # Convert list of dicts to DataFrame
            df = pd.DataFrame(bars)
            if not df.empty:
                # Ensure columns are lowercase
                df.columns = [col.lower() for col in df.columns]
                # Sort by date
                if "date" in df.columns:
                    df = df.sort_values("date")
        else:
            # Legacy IB client - use synchronous methods
            if not self.ib.isConnected():
                raise ConnectionError("Not connected to IBKR")

            contract = Stock(symbol, "SMART", "USD")
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return pd.DataFrame()

            bars = self.ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df = df.sort_values("date")

        return df

    async def teardown(self, full_cleanup: bool = False):
        """Clean up resources after a run cycle.

        Args:
            full_cleanup: If True, calls full cleanup() including IBKR disconnect.
                          If False (default), only stops monitors but keeps connection alive.
        """
        # Only stop monitors that need cycle-level cleanup
        if self.production_monitor:
            await self.production_monitor.stop()
        if self.correlation_manager:
            await self.correlation_manager.stop()

        # Only do full cleanup (IBKR disconnect) when explicitly requested
        # This allows persistent connections across trading cycles
        if full_cleanup:
            await self.cleanup()
        else:
            logger.info("Cycle complete - keeping IBKR connection alive for next cycle")

    async def _monitor_subprocess_health(self):
        """
        Background task to monitor subprocess health.

        Pings the subprocess every 60 seconds and automatically restarts
        if the subprocess becomes unresponsive after multiple failures.
        """
        logger.info("Starting subprocess health monitoring (60s interval, 3 failures to restart)")
        consecutive_failures = 0
        max_failures = 3  # Require 3 consecutive failures before restart

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Only monitor if using subprocess client
                if not hasattr(self.ib, "ping"):
                    logger.debug("Not using subprocess client, skipping health check")
                    continue

                # Ping subprocess
                logger.debug("Pinging subprocess for health check...")
                is_healthy = await self.ib.ping()

                if not is_healthy:
                    consecutive_failures += 1
                    logger.warning(
                        f"Subprocess health check failed ({consecutive_failures}/{max_failures})"
                    )

                    if consecutive_failures >= max_failures:
                        logger.error(
                            f"Subprocess unresponsive after {max_failures} consecutive failures - restarting"
                        )
                        await self._restart_subprocess()
                        consecutive_failures = 0  # Reset after restart attempt
                else:
                    if consecutive_failures > 0:
                        logger.info("Subprocess health check recovered")
                    consecutive_failures = 0
                    logger.debug("Subprocess health check passed")

            except asyncio.CancelledError:
                logger.info("Subprocess health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subprocess health monitoring: {e}")
                consecutive_failures += 1
                # Continue monitoring even on errors
                continue

    async def _restart_subprocess(self):
        """
        Restart the IBKR subprocess after a crash or health check failure.

        This attempts to gracefully stop the subprocess and create a new
        connection using the same configuration.
        """
        logger.warning("⚠️ Restarting IBKR subprocess due to health check failure")

        try:
            # Clean up stale lock files first
            import os

            lock_path = os.environ.get(
                "IBKR_LOCK_FILE_PATH", "/tmp/ibkr_connect.lock"
            )  # nosec B108
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                    logger.info(f"Removed stale lock file: {lock_path}")
            except Exception as e:
                logger.warning(f"Could not remove lock file: {e}")

            # Stop the old subprocess
            if hasattr(self.ib, "stop"):
                try:
                    await asyncio.wait_for(self.ib.stop(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Subprocess stop timed out, continuing with restart")
                except Exception as e:
                    logger.warning(f"Error stopping subprocess: {e}")

            # Kill any orphaned worker processes
            import subprocess

            subprocess.run(
                ["pkill", "-9", "-f", "ibkr_subprocess_worker"], capture_output=True, timeout=5
            )
            await asyncio.sleep(1)  # Give time for cleanup

            # Reconnect using robust connection
            from .utils.robust_connection import CircuitBreakerConfig, connect_ibkr_robust

            # Get connection parameters from config
            host = self.cfg.ibkr.host
            port = self.cfg.ibkr.port

            circuit_config = CircuitBreakerConfig(
                failure_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
                recovery_timeout=float(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300")),
                success_threshold=2,
            )

            # Reconnect
            logger.info(f"Reconnecting to IBKR at {host}:{port}")
            self.ib = await connect_ibkr_robust(
                host=host,
                port=port,
                client_id=self.cfg.ibkr.client_id,
                readonly=self.cfg.ibkr.readonly,
                timeout=self.cfg.ibkr.timeout,
                max_retries=3,  # More retries
                circuit_breaker_config=circuit_config,
                ssl_mode=self.cfg.ibkr.ssl_mode,
            )

            logger.info("✅ Subprocess restarted successfully")

        except Exception as e:
            logger.error(f"❌ Failed to restart subprocess: {e}")
            logger.warning("Will retry on next health check cycle")
            # Don't raise - let it retry on the next health check
            # raise RuntimeError(f"Failed to restart subprocess: {e}")

    async def fetch_and_store_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data for a symbol and store in database."""
        # Check if market is open before fetching data
        trading_allowed = is_trading_allowed()
        if not trading_allowed:
            session = get_market_session()
            logger.info(
                f"⏸️ Market is {session}, trading_allowed={trading_allowed}, ENABLE_EXTENDED_HOURS={ENABLE_EXTENDED_HOURS}, skipping data fetch for {symbol}"
            )
            return None

        # Check connection health and reconnect if needed
        if hasattr(self.ib, "is_connected") and not self.ib.is_connected:
            logger.warning(f"Connection lost before fetching {symbol}, attempting reconnect...")
            try:
                await self.restart_subprocess()
            except Exception as e:
                logger.error(f"Failed to reconnect: {e}")
                return None

        try:
            start_time = asyncio.get_event_loop().time()
            with Timer("data_fetch", self.monitor):
                # Fetch historical bars using IB connection directly
                df = await self._fetch_historical_bars(
                    symbol=symbol, duration=self.duration, bar_size=self.bar_size
                )

            # Record API call metrics to ProductionMonitor (Phase 4 P2)
            if self.production_monitor:
                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                self.production_monitor.record_api_call(
                    "fetch_historical_bars", df is not None and not df.empty, latency_ms
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
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()
                batch_data.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": int(row.get("volume", 0)),
                    }
                )

            if batch_data:
                with Timer("database_write", self.monitor):
                    await self.db.batch_store_market_data(batch_data)
                self.monitor.record_data_points(len(batch_data))

            return df

        except Exception as e:
            error_str = str(e)
            logger.error(f"Failed to fetch data for {symbol}: {e}")

            # If connection error, mark as disconnected to trigger reconnect on next fetch
            if "ConnectionError" in error_str or "Not connected" in error_str:
                if hasattr(self.ib, "_connected"):
                    self.ib._connected = False
                logger.warning("Marked connection as disconnected for next retry")

            return None

    def _manage_cache_size(self):
        """Ensure cache doesn't exceed max size (LRU eviction)."""
        while len(self.market_data_cache) > self.max_cache_size:
            # Remove oldest item (first item in OrderedDict)
            oldest_symbol = next(iter(self.market_data_cache))
            del self.market_data_cache[oldest_symbol]
            logger.debug(f"Evicted {oldest_symbol} from market data cache")

    def _blocked_result(
        self,
        symbol: str,
        signal: int,
        price: float,
        message: str,
        df: Optional[pd.DataFrame] = None,
    ) -> SymbolResult:
        """Create a SymbolResult for blocked/skipped trades.

        Centralizes the creation of "no action" results to reduce code duplication.
        """
        return SymbolResult(
            symbol=symbol,
            signal=signal,
            price=price,
            quantity=0,
            executed=False,
            message=message,
            data=df,
        )

    async def process_symbol(self, symbol: str) -> SymbolResult:
        """Process a single symbol - fetch data, generate signal, execute if needed."""
        # Fetch and store market data
        df = await self.fetch_and_store_data(symbol)
        if df is None or df.empty:
            return SymbolResult(
                symbol=symbol,
                signal=0,
                price=0.0,
                quantity=0,
                executed=False,
                message="No data available",
            )

        # Cache market data for pairs/stat arb analysis with LRU eviction
        if symbol in self.market_data_cache:
            self.market_data_cache.move_to_end(symbol)
        self.market_data_cache[symbol] = df
        self._manage_cache_size()

        # Send real-time price update via WebSocket
        latest_price = None
        if df is not None and not df.empty:
            latest_price = float(df["close"].iloc[-1])
            self.latest_prices[symbol] = latest_price

            if WEBSOCKET_ENABLED and ws_client:
                try:
                    ws_client.send_market_update(symbol, latest_price)
                    logger.info(f"Sent WebSocket update for {symbol}: ${latest_price:.2f}")
                except Exception as e:
                    logger.error(f"Could not send WebSocket update: {e}")

            # Update advanced risk manager with latest prices
            if self.use_advanced_risk and self.advanced_risk and latest_price:
                self.advanced_risk.update_market_prices({symbol: latest_price})

            # Update stop-loss monitor with latest price
            if self.stop_loss_monitor and latest_price:
                await self.stop_loss_monitor.update_price(symbol, latest_price)

        # Generate trading signal
        with Timer("signal_generation", self.monitor):
            if self.use_ml_enhanced and self.ml_enhanced_strategy:
                # Use ML Enhanced strategy for signal generation
                signal_obj = await self.ml_enhanced_strategy.analyze(symbol, df)

                # Convert ML Enhanced signal to format compatible with rest of code
                if signal_obj:
                    signal_value = (
                        1
                        if signal_obj.action == "BUY"
                        else (-1 if signal_obj.action == "SELL" else 0)
                    )
                    confidence = signal_obj.confidence
                    # Extract additional features if available
                    position_size = signal_obj.features.get("position_size", 0.02)
                    stop_loss = signal_obj.features.get("stop_loss", 0.02)
                    take_profit = signal_obj.features.get("take_profit", 0.05)

                    # Track prediction for dashboard
                    self._ml_predictions[symbol] = {
                        "signal": signal_value,
                        "confidence": confidence,
                        "action": signal_obj.action,
                        "source": "ML_ENHANCED",
                        "timestamp": datetime.now().isoformat(),
                    }
                else:
                    # ML returned no signal - track as HOLD and use AI analyst fallback
                    signal_value = 0
                    confidence = 0.5
                    position_size = 0.02
                    stop_loss = 0.02
                    take_profit = 0.05

                    # Track ML HOLD for dashboard
                    self._ml_predictions[symbol] = {
                        "signal": 0,
                        "confidence": 0.5,
                        "action": "HOLD",
                        "source": "ML_NO_SIGNAL",
                        "timestamp": datetime.now().isoformat(),
                    }

                    # Check if this symbol is an AI-identified opportunity (from news scan)
                    ai_opps = getattr(self, "_ai_opportunities", {})
                    if symbol in ai_opps:
                        opp = ai_opps[symbol]
                        signal_value = 1  # BUY
                        confidence = opp.get("confidence", 0.7)
                        logger.info(
                            f"🎯 AI OPPORTUNITY BUY for {symbol}: "
                            f"conf={confidence:.0%} - {opp.get('reason', 'AI identified')}"
                        )
                    # Try AI-driven signal from news analysis
                    elif self.ai_analyst and AI_ANALYST_AVAILABLE:
                        try:
                            # Fetch news (cached for 5 minutes)
                            from datetime import timedelta

                            now = datetime.now()
                            if self.last_news_fetch is None or (
                                now - self.last_news_fetch
                            ) > timedelta(minutes=5):
                                self.news_cache = {
                                    item["title"]: item for item in fetch_rss_news(max_items=15)
                                }
                                self.last_news_fetch = now
                                logger.info(f"Fetched {len(self.news_cache)} news items")

                            # Find relevant news for this symbol
                            symbol_news = [
                                n
                                for n in self.news_cache.values()
                                if symbol.upper() in n["title"].upper()
                            ]

                            # Also check general market news for broader context
                            market_keywords = [
                                "market",
                                "fed",
                                "economy",
                                "inflation",
                                "stocks",
                                "rally",
                                "crash",
                            ]
                            market_news = [
                                n
                                for n in self.news_cache.values()
                                if any(kw in n["title"].lower() for kw in market_keywords)
                            ]

                            relevant_news = symbol_news or market_news[:3]

                            if relevant_news:
                                # Combine news into single event text
                                news_text = "\n".join(
                                    [f"- {n['title']} ({n['source']})" for n in relevant_news[:3]]
                                )

                                # Get AI analysis
                                analysis = self.ai_analyst.analyze_market_event(
                                    symbol=symbol,
                                    event_text=news_text,
                                    market_data={
                                        "price": float(df["close"].iloc[-1]) if len(df) > 0 else 0,
                                        "change_5d": (
                                            float(
                                                (df["close"].iloc[-1] / df["close"].iloc[0] - 1)
                                                * 100
                                            )
                                            if len(df) > 5
                                            else 0
                                        ),
                                    },
                                )

                                # Convert AI suggestion to signal
                                if analysis.suggested_action == "buy" and analysis.confidence > 0.5:
                                    signal_value = 1
                                    confidence = analysis.confidence
                                    logger.info(
                                        f"AI BUY signal for {symbol}: {analysis.reasoning[:80]}"
                                    )
                                    # Update prediction with AI signal
                                    self._ml_predictions[symbol] = {
                                        "signal": 1,
                                        "confidence": confidence,
                                        "action": "BUY",
                                        "source": "AI_ANALYST",
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                elif (
                                    analysis.suggested_action == "sell"
                                    and analysis.confidence > 0.5
                                ):
                                    signal_value = -1
                                    confidence = analysis.confidence
                                    logger.info(
                                        f"AI SELL signal for {symbol}: {analysis.reasoning[:80]}"
                                    )
                                    # Update prediction with AI signal
                                    self._ml_predictions[symbol] = {
                                        "signal": -1,
                                        "confidence": confidence,
                                        "action": "SELL",
                                        "source": "AI_ANALYST",
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                else:
                                    logger.debug(
                                        f"AI HOLD for {symbol}: {analysis.suggested_action} conf={analysis.confidence:.2f}"
                                    )
                        except Exception as e:
                            logger.warning(f"AI analysis failed for {symbol}: {e}")

                    # Fall back to SMA if AI didn't produce a signal
                    if signal_value == 0:
                        sma_signals = sma_crossover_signals(
                            pd.DataFrame({"close": df["close"]}),
                            fast=self.sma_fast,
                            slow=self.sma_slow,
                        )
                        signal_value = (
                            int(sma_signals["signal"].iloc[-1]) if len(sma_signals) > 0 else 0
                        )
                        if signal_value != 0:
                            confidence = 0.6
                            logger.info(f"SMA crossover signal for {symbol}: {signal_value}")

                signals = pd.DataFrame(
                    {
                        "signal": [signal_value],
                        "confidence": [confidence],
                        "position_size": [position_size],
                        "stop_loss": [stop_loss],
                        "take_profit": [take_profit],
                    },
                    index=[df.index[-1]] if len(df) > 0 else [pd.Timestamp.now()],
                )
            elif self.use_ml_strategy and self.ml_strategy:
                # Use ML strategy for signal generation
                # Prepare market data in the format expected by ML strategy
                market_data_dict = {
                    "symbol": symbol,
                    "data": df,
                    "price": float(df["close"].iloc[-1]) if len(df) > 0 else 0,
                    "portfolio_value": 100000,  # Will be replaced with actual portfolio value
                    "atr": df["close"].rolling(14).std().iloc[-1] if len(df) > 14 else 2,
                }

                signal_obj = await self.ml_strategy.generate_signal(
                    symbol=symbol, market_data=market_data_dict
                )

                # Convert ML signal to format compatible with rest of code
                if signal_obj:
                    signal_value = (
                        1
                        if signal_obj.action == "BUY"
                        else (-1 if signal_obj.action == "SELL" else 0)
                    )
                    confidence = signal_obj.confidence
                else:
                    signal_value = 0
                    confidence = 0.5

                signals = pd.DataFrame(
                    {
                        "close": [df["close"].iloc[-1]],
                        "signal": [signal_value],
                        "confidence": [confidence],
                    }
                )
            else:
                # Use traditional SMA crossover strategy
                signals = sma_crossover_signals(
                    pd.DataFrame({"close": df["close"]}),
                    fast=self.sma_fast,
                    slow=self.sma_slow,
                )

        # Generate mean reversion signals if enabled
        if MEAN_REVERSION_AVAILABLE and self.mean_reversion_strategy:
            try:
                from .features.engine import FeatureSet

                # Create basic feature set (can be enhanced)
                feature_set = FeatureSet(timestamp=pd.Timestamp.now(tz="UTC"), symbol=symbol)
                if len(df) >= 20:
                    # Calculate basic features
                    feature_set.bb_upper = (
                        df["close"].rolling(20).mean().iloc[-1]
                        + 2 * df["close"].rolling(20).std().iloc[-1]
                    )
                    feature_set.bb_middle = df["close"].rolling(20).mean().iloc[-1]
                    feature_set.bb_lower = (
                        df["close"].rolling(20).mean().iloc[-1]
                        - 2 * df["close"].rolling(20).std().iloc[-1]
                    )
                    feature_set.atr = (
                        df["close"].rolling(14).std().iloc[-1]
                        if len(df) >= 14
                        else df["close"].iloc[-1] * 0.02
                    )

                    # Simple RSI calculation
                    delta = df["close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    feature_set.rsi = (
                        100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
                    )

                # Generate mean reversion signals
                mean_rev_signals = await self.mean_reversion_strategy._generate_signals(
                    {symbol: df}, {symbol: feature_set}
                )

                # Log mean reversion signals
                if mean_rev_signals:
                    for mr_signal in mean_rev_signals:
                        logger.info(
                            "mean_reversion_signal",
                            symbol=symbol,
                            signal_type=mr_signal.signal_type.value,
                            strength=mr_signal.strength,
                            reversion_score=mr_signal.metadata.get("reversion_score", 0),
                        )

                        # Override signal if mean reversion is stronger
                        if mr_signal.strength > 0.7:
                            signal_value = 1 if mr_signal.signal_type.value == "BUY" else -1
                            signals.iloc[-1]["signal"] = signal_value

            except Exception as e:
                logger.error(f"Mean reversion signal generation error for {symbol}: {e}")

        # Update correlation tracker with price data if enabled
        if self.use_correlation_sizing and self.correlation_tracker:
            # Add price series for correlation calculation with sector classification
            sector = self._get_symbol_sector(symbol)
            self.correlation_tracker.add_price_series(
                symbol=symbol, prices=df["close"], sector=sector
            )

        last = signals.iloc[-1]
        price = PrecisePricing.to_decimal(last.get("close", df["close"].iloc[-1]))
        price_float = float(price)
        self.latest_prices[symbol] = price_float

        equity_prices = {
            sym: self.latest_prices.get(sym, self.positions[sym].avg_price)
            for sym in self.positions
        }
        equity_prices[symbol] = price_float
        # Get current equity (thread-safe)
        equity = await self.portfolio.equity(equity_prices)

        # Record signal in database with strength reflecting model confidence if available
        signal_value = int(last.get("signal", 0))
        signal_strength = float(last.get("confidence", 1.0)) if "confidence" in last else 1.0

        # Determine signal type for WebSocket/logging
        if signal_value == 1:
            signal_type = "BUY"
        elif signal_value == -1:
            signal_type = "SELL"
        else:
            signal_type = "HOLD"

        # Record non-zero signals in database
        if signal_value != 0:
            strategy_name = (
                "ML_ENHANCED"
                if self.use_ml_enhanced
                else "ML_ENSEMBLE"
                if self.use_ml_strategy
                else "SMA_CROSSOVER"
            )
            await self.db.record_signal(
                symbol,
                strategy_name,
                "BUY" if signal_value == 1 else "SELL",
                max(0.0, min(1.0, float(signal_strength))),
            )

        # ALWAYS send signal update via WebSocket (including HOLD signals)
        # This lets the dashboard show real-time activity
        if WEBSOCKET_ENABLED and ws_client:
            try:
                ws_client.send_signal_update(
                    symbol,
                    signal_type,
                    max(0.0, min(1.0, float(signal_strength))),
                    reason=f"Price: ${price_float:.2f}",
                )
            except Exception as e:
                logger.debug(f"Could not send signal WebSocket update: {e}")

        # Execute trades based on signal
        executed = False
        message = "No action"
        quantity = 0

        if signal_value == 1:  # Buy signal
            # FIRST CHECK: Has this symbol already had a BUY attempted/executed this cycle?
            # Use combined lock check for maximum protection
            async with self._cycle_executed_buys_lock:
                if symbol in self._cycle_executed_buys:
                    logger.warning(
                        f"DUPLICATE BUY BLOCKED: {symbol} already had BUY attempted this cycle"
                    )
                    return self._blocked_result(
                        symbol,
                        signal_value,
                        price_float,
                        "Duplicate buy blocked: already attempted this cycle",
                        df,
                    )
                # CRITICAL: Mark as attempted IMMEDIATELY to prevent any other task from proceeding
                self._cycle_executed_buys.add(symbol)
                logger.info(f"Marked {symbol} for BUY processing this cycle")

            if symbol in self.positions and self.positions[symbol].quantity < 0:
                # Cover short position first
                pos = self.positions[symbol]
                qty_to_cover = abs(pos.quantity)

                # Check kill switch before order execution
                if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                    if self.advanced_risk.kill_switch.triggered:
                        logger.error(
                            f"KILL SWITCH ACTIVE - Order blocked for {symbol} BUY_TO_COVER"
                        )
                        return self._blocked_result(
                            symbol, 0, price_float, "Kill switch active - trading halted", df
                        )

                res = await self._place_order_with_circuit_breaker(
                    Order(symbol=symbol, quantity=qty_to_cover, side="BUY_TO_COVER", price=price)
                )
                if res.ok:
                    if self.stop_loss_monitor:
                        self.stop_loss_monitor.cancel_stop(symbol)
                    fill_price = res.fill_price if res.fill_price is not None else price_float
                    success = await self._update_position_atomic(
                        symbol, qty_to_cover, fill_price, "BUY_TO_COVER"
                    )
                    if success:
                        self.daily_pnl = float(self.portfolio.realized_pnl)

                        await self.db.record_trade(
                            symbol,
                            "BUY_TO_COVER",
                            qty_to_cover,
                            fill_price,
                            slippage=(
                                (fill_price - price_float) * qty_to_cover
                                if res.fill_price is not None
                                else 0
                            ),
                        )
                        await self.db.update_position(symbol, 0, 0, 0)  # Close position

                        self.monitor.record_order_placed(symbol, qty_to_cover)
                        self.monitor.record_trade_executed(symbol, "BUY_TO_COVER", qty_to_cover)

                        if self.production_monitor:
                            latency_ms = 10  # Simulated latency for paper trading
                            self.production_monitor.record_order(symbol, True, latency_ms)
                            pnl = (pos.avg_cost - fill_price) * qty_to_cover
                            self.production_monitor.record_trade(symbol, pnl, True)

                        executed = True
                        quantity = qty_to_cover
                        message = (
                            f"Covered short: Bought {qty_to_cover} shares at ${fill_price:.2f}"
                        )
                    else:
                        logger.error(f"Failed to update position for {symbol} BUY_TO_COVER order")
                        message = f"Cover order failed: atomic update error"
                else:
                    message = f"Cover order failed: {res.msg}"

            elif symbol not in self.positions:
                # CRITICAL: Check for pending orders to prevent duplicate buys
                # This prevents race conditions when processing symbols in parallel
                async with self._pending_orders_lock:
                    if symbol in self._pending_orders:
                        logger.warning(f"DUPLICATE BUY BLOCKED: {symbol} already has pending order")
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            "Duplicate buy blocked: order already pending",
                            df,
                        )
                    # Re-check position inside lock to prevent TOCTOU race
                    if symbol in self.positions:
                        logger.info(
                            f"Position already exists for {symbol} (race condition prevented)"
                        )
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            "Buy signal: Already have long position",
                            df,
                        )
                    # Mark as pending BEFORE releasing lock
                    self._pending_orders.add(symbol)
                    logger.debug(f"Added {symbol} to pending orders")

                    # CRITICAL: Check DATABASE for existing position (prevents cross-cycle duplicates)
                    db_positions = await self.db.get_positions()
                    db_position = next(
                        (
                            p
                            for p in db_positions
                            if p.get("symbol") == symbol and p.get("quantity", 0) > 0
                        ),
                        None,
                    )
                    if db_position:
                        logger.warning(
                            f"DUPLICATE BUY BLOCKED: {symbol} already has DB position (qty={db_position.get('quantity')})"
                        )
                        self._pending_orders.discard(symbol)
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            f"Duplicate buy blocked: DB position exists (qty={db_position.get('quantity')})",
                            df,
                        )

                    # ADDITIONAL CHECK: Look for recent BUY trades (handles race conditions)
                    # Position might not be updated yet if trade just happened
                    recent_buy = await self.db.has_recent_buy_trade(symbol, seconds=600)
                    if recent_buy:
                        logger.warning(
                            f"DUPLICATE BUY BLOCKED: {symbol} has recent BUY trade in last 10 minutes"
                        )
                        self._pending_orders.discard(symbol)
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            "Duplicate buy blocked: recent BUY trade exists",
                            df,
                        )

                    # ANTI-CHURN CHECK: Block re-buying a symbol recently sold
                    # Prevents rapid BUY→SELL→BUY cycling (whipsawing)
                    recent_sell = await self.db.has_recent_sell_trade(symbol, seconds=600)
                    if recent_sell:
                        logger.warning(
                            f"CHURN BLOCKED: {symbol} was sold in last 10 minutes - waiting for cooldown"
                        )
                        self._pending_orders.discard(symbol)
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            "Churn blocked: recently sold - 10 min cooldown",
                            df,
                        )

                try:
                    # Open long position
                    # Use advanced risk manager with Kelly sizing if enabled
                    if self.use_advanced_risk and self.advanced_risk:
                        # Get ATR for stop loss calculation (if available)
                        atr = df["atr"].iloc[-1] if "atr" in df.columns else None

                        # Calculate position size using Kelly criterion
                        sizing_result = await self.advanced_risk.calculate_position_size(
                            symbol=symbol,
                            signal_strength=abs(signal_value),  # Use signal strength from strategy
                            current_price=price_float,  # Use float, not Decimal
                            atr=atr,
                        )

                        if sizing_result["blocked"]:
                            logger.warning(
                                f"Trade blocked by kill switch for {symbol}: {sizing_result['block_reason']}"
                            )
                            return self._blocked_result(
                                symbol,
                                signal_value,
                                price_float,
                                f"Blocked: {sizing_result['block_reason']}",
                                df,
                            )

                        qty = sizing_result["position_size"]

                        # Log Kelly metrics
                        if "kelly_metrics" in sizing_result:
                            km = sizing_result["kelly_metrics"]
                            logger.info(
                                f"Kelly metrics for {symbol}: Win rate={km['win_rate']:.2%}, Edge={km['edge']:.3f}, Kelly fraction={sizing_result['kelly_fraction']:.3f}"
                            )

                        # Log any warnings
                        for warning in sizing_result.get("warnings", []):
                            logger.warning(f"Risk warning for {symbol}: {warning}")
                    else:
                        # Fallback to standard position sizing
                        qty = self.risk.position_size(equity, price)

                    # Scale by portfolio manager strategy weight if available
                    if self.portfolio_manager and self.active_strategy_name:
                        alloc = self.portfolio_manager.allocations.get(self.active_strategy_name)
                        if alloc:
                            qty = max(int(qty * alloc.current_weight), 0)

                    # Apply correlation-based position sizing if enabled
                    if self.use_correlation_sizing and self.correlation_manager:
                        (
                            adjusted_qty,
                            sizing_reason,
                        ) = await self.correlation_manager.get_adjusted_position_size(
                            symbol=symbol,
                            base_size=qty,
                            current_positions=self.positions,
                            portfolio_value=equity,
                        )
                        if adjusted_qty != qty:
                            logger.info(
                                f"Position size adjusted for {symbol}: {qty} -> {adjusted_qty} ({sizing_reason})"
                            )
                            qty = adjusted_qty

                    # Enforce minimum position value to prevent position sprawl
                    min_position_value = float(os.getenv("MIN_POSITION_VALUE", "500"))
                    position_value = qty * price_float
                    if position_value < min_position_value and qty > 0:
                        logger.info(
                            f"Skipping {symbol}: position value ${position_value:.0f} below minimum ${min_position_value:.0f}"
                        )
                        return self._blocked_result(
                            symbol,
                            signal_value,
                            price_float,
                            f"Position value ${position_value:.0f} below min ${min_position_value:.0f}",
                            df,
                        )

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
                        # Check kill switch before order execution
                        if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                            if self.advanced_risk.kill_switch.triggered:
                                logger.error(f"KILL SWITCH ACTIVE - Order blocked for {symbol} BUY")
                                return self._blocked_result(
                                    symbol,
                                    0,
                                    price_float,
                                    "Kill switch active - trading halted",
                                    df,
                                )

                        res = await self._place_order_with_circuit_breaker(
                            Order(symbol=symbol, quantity=qty, side="BUY", price=price)
                        )
                        if res.ok:
                            fill_price = (
                                res.fill_price if res.fill_price is not None else price_float
                            )
                            # Use atomic position update to prevent race conditions
                            success = await self._update_position_atomic(
                                symbol, qty, fill_price, "BUY"
                            )
                            if success:
                                self.daily_executed_notional += price_float * qty

                                # Record trade in database
                                await self.db.record_trade(
                                    symbol,
                                    "BUY",
                                    qty,
                                    fill_price,
                                    slippage=(
                                        (fill_price - price_float) * qty
                                        if res.fill_price is not None
                                        else 0
                                    ),
                                )
                                # Use accumulated position qty/avg from self.positions, not just this order's qty
                                pos = self.positions[symbol]
                                await self.db.update_position(
                                    symbol, pos.quantity, pos.avg_price, price
                                )

                                self.monitor.record_order_placed(symbol, qty)
                                self.monitor.record_trade_executed(symbol, "BUY", qty)

                                # Record metrics to ProductionMonitor (Phase 4 P2)
                                if self.production_monitor:
                                    latency_ms = 10  # Simulated latency for paper trading
                                    self.production_monitor.record_order(symbol, True, latency_ms)
                                    # No PnL yet for opening position
                                    self.production_monitor.record_trade(symbol, 0, True)

                                executed = True
                                quantity = qty
                                message = f"Opened long: Bought {qty} shares at ${fill_price:.2f}"

                                # Add stop-loss order for the new position
                                if self.stop_loss_monitor and self.enable_stop_loss:
                                    try:
                                        # Create position object for stop-loss
                                        new_position = Position(
                                            symbol=symbol,
                                            quantity=qty,
                                            avg_price=fill_price,
                                            entry_time=datetime.now(),
                                        )
                                        if self.use_trailing_stop:
                                            await self.stop_loss_monitor.add_stop_loss(
                                                symbol=symbol,
                                                position=new_position,
                                                stop_percent=self.trailing_stop_pct,
                                                stop_type=StopType.TRAILING_PERCENT,
                                                trailing_percent=self.trailing_stop_pct,
                                            )
                                            logger.info(
                                                f"Trailing stop added for {symbol} at {self.trailing_stop_pct:.1%}"
                                            )
                                        else:
                                            await self.stop_loss_monitor.add_stop_loss(
                                                symbol=symbol,
                                                position=new_position,
                                                stop_percent=self.stop_loss_percent,
                                                stop_type=StopType.FIXED,
                                            )
                                            logger.info(
                                                f"Fixed stop-loss added for {symbol} at {self.stop_loss_percent:.1%}"
                                            )
                                    except Exception as e:
                                        logger.error(f"Failed to add stop-loss for {symbol}: {e}")

                                # Send trade update via WebSocket
                                if WEBSOCKET_ENABLED and ws_client:
                                    try:
                                        ws_client.send_trade_update(symbol, "BUY", qty, fill_price)
                                    except Exception as e:
                                        logger.debug(f"Could not send trade WebSocket update: {e}")
                            else:
                                logger.error(f"Failed to update position for {symbol} BUY order")
                                message = f"Buy order failed: atomic update error"
                        else:
                            message = f"Buy order failed: {res.msg}"
                    else:
                        message = f"Buy signal rejected: {msg}"
                finally:
                    # CRITICAL: Always remove from pending orders when done
                    async with self._pending_orders_lock:
                        self._pending_orders.discard(symbol)
                        logger.debug(f"Removed {symbol} from pending orders")
            else:
                message = "Buy signal: Already have long position"

        elif signal_value == -1:  # Sell signal
            enable_short_selling = self.cfg.execution.enable_short_selling
            logger.info(
                f"SELL signal for {symbol}: checking position (have position: {symbol in self.positions})"
            )

            if symbol in self.positions:
                # Close long position or cover short
                pos = self.positions[symbol]

                if pos.quantity > 0:  # Closing long position
                    # PRO TRADER RULE: Only sell on signal if position is profitable
                    # If at a loss, let the stop-loss handle the exit (don't panic sell)
                    entry_price = float(pos.avg_price)

                    # Guard against division by zero (corrupt data edge case)
                    if entry_price <= 0:
                        logger.error(
                            f"Invalid entry_price for {symbol}: {entry_price} - skipping sell check"
                        )
                        return self._blocked_result(
                            symbol, 0, price_float, f"Invalid entry price: {entry_price}", df
                        )

                    pnl_percent = ((price_float - entry_price) / entry_price) * 100

                    if price_float < entry_price:
                        # Position is at a loss - don't sell on ML signal
                        loss_amount = (entry_price - price_float) * pos.quantity
                        message = (
                            f"SELL signal IGNORED for {symbol}: position at loss "
                            f"(entry ${entry_price:.2f}, current ${price_float:.2f}, "
                            f"P&L {pnl_percent:.1f}% = -${loss_amount:.2f}). "
                            f"Letting stop-loss handle exit."
                        )
                        logger.warning(message)
                        return self._blocked_result(symbol, 0, price_float, message, df)

                    logger.info(
                        f"Attempting to SELL {pos.quantity} shares of {symbol} at ${price_float:.2f} "
                        f"(profit: {pnl_percent:.1f}%)"
                    )

                    # Check kill switch before order execution
                    if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                        if self.advanced_risk.kill_switch.triggered:
                            logger.error(f"KILL SWITCH ACTIVE - Order blocked for {symbol} SELL")
                            return self._blocked_result(
                                symbol, 0, price_float, "Kill switch active - trading halted", df
                            )

                    res = await self._place_order_with_circuit_breaker(
                        Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=price)
                    )
                    if res.ok:
                        # Cancel stop-loss when closing position
                        if self.stop_loss_monitor:
                            self.stop_loss_monitor.cancel_stop(symbol)
                        # Use price_float for consistency (price is Decimal, fill_price is float)
                        fill_price = res.fill_price or price_float
                        # Use atomic position update to prevent race conditions
                        success = await self._update_position_atomic(
                            symbol, pos.quantity, fill_price, "SELL"
                        )
                        if success:
                            self.daily_pnl = float(self.portfolio.realized_pnl)

                            # Record trade in database
                            await self.db.record_trade(
                                symbol,
                                "SELL",
                                pos.quantity,
                                fill_price,
                                slippage=(
                                    (float(fill_price) - float(price)) * pos.quantity
                                    if res.fill_price
                                    else 0
                                ),
                            )
                            await self.db.update_position(symbol, 0, 0, 0)  # Close position

                            self.monitor.record_order_placed(symbol, pos.quantity)
                            self.monitor.record_trade_executed(symbol, "SELL", pos.quantity)

                            # Record metrics to ProductionMonitor (Phase 4 P2)
                            if self.production_monitor:
                                latency_ms = 10  # Simulated latency for paper trading
                                self.production_monitor.record_order(symbol, True, latency_ms)
                                pnl = (
                                    float(fill_price) - float(pos.avg_price)
                                ) * pos.quantity  # Long position PnL
                                self.production_monitor.record_trade(symbol, pnl, pnl > 0)

                            executed = True
                            quantity = pos.quantity
                            message = (
                                f"Closed long: Sold {pos.quantity} shares at ${fill_price:.2f}"
                            )
                        else:
                            logger.error(f"Failed to update position for {symbol} SELL order")
                            message = f"Sell order failed: atomic update error"
                    else:
                        message = f"Sell order failed: {res.msg}"

            elif enable_short_selling and symbol not in self.positions:
                # Open short position
                qty = self.risk.position_size(equity, price)

                # Scale by portfolio manager strategy weight if available
                if self.portfolio_manager and self.active_strategy_name:
                    alloc = self.portfolio_manager.allocations.get(self.active_strategy_name)
                    if alloc:
                        qty = max(int(qty * alloc.current_weight), 0)

                # Apply correlation-based position sizing if enabled
                if self.use_correlation_sizing and self.correlation_manager:
                    (
                        adjusted_qty,
                        sizing_reason,
                    ) = await self.correlation_manager.get_adjusted_position_size(
                        symbol=symbol,
                        base_size=qty,
                        current_positions=self.positions,
                        portfolio_value=equity,
                    )
                    if adjusted_qty != qty:
                        logger.info(
                            f"Short size adjusted for {symbol}: {qty} -> {adjusted_qty} ({sizing_reason})"
                        )
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
                    # Check kill switch before order execution
                    if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                        if self.advanced_risk.kill_switch.triggered:
                            logger.error(
                                f"KILL SWITCH ACTIVE - Order blocked for {symbol} SELL_SHORT"
                            )
                            return self._blocked_result(
                                symbol, 0, price_float, "Kill switch active - trading halted", df
                            )

                    res = await self._place_order_with_circuit_breaker(
                        Order(symbol=symbol, quantity=qty, side="SELL_SHORT", price=price)
                    )
                    if res.ok:
                        fill_price = res.fill_price if res.fill_price is not None else price_float

                        if self.stop_loss_monitor and self.enable_stop_loss:
                            try:
                                short_position = Position(
                                    symbol=symbol,
                                    quantity=-qty,
                                    avg_price=fill_price,
                                    entry_time=datetime.now(),
                                )
                                if self.use_trailing_stop:
                                    await self.stop_loss_monitor.add_stop_loss(
                                        symbol=symbol,
                                        position=short_position,
                                        stop_percent=self.trailing_stop_pct,
                                        stop_type=StopType.TRAILING_PERCENT,
                                        trailing_percent=self.trailing_stop_pct,
                                    )
                                    logger.info(
                                        f"Trailing stop added for SHORT {symbol} at {self.trailing_stop_pct:.1%}"
                                    )
                                else:
                                    await self.stop_loss_monitor.add_stop_loss(
                                        symbol=symbol,
                                        position=short_position,
                                        stop_percent=self.stop_loss_percent,
                                        stop_type=StopType.FIXED,
                                    )
                                    logger.info(
                                        f"Fixed stop-loss added for SHORT {symbol} at {self.stop_loss_percent:.1%}"
                                    )
                            except Exception as e:
                                logger.error(f"Failed to add stop-loss for short {symbol}: {e}")

                        success = await self._update_position_atomic(
                            symbol, qty, fill_price, "SELL_SHORT"
                        )
                        if success:
                            self.daily_executed_notional += price_float * qty

                            await self.db.record_trade(
                                symbol,
                                "SELL_SHORT",
                                qty,
                                fill_price,
                                slippage=(
                                    (fill_price - price_float) * qty
                                    if res.fill_price is not None
                                    else 0
                                ),
                            )
                            # Use accumulated position qty/avg from self.positions
                            pos = self.positions[symbol]
                            await self.db.update_position(
                                symbol, pos.quantity, pos.avg_price, price
                            )

                            self.monitor.record_order_placed(symbol, qty)
                            self.monitor.record_trade_executed(symbol, "SELL_SHORT", qty)
                            executed = True
                            quantity = qty
                            message = f"Opened short: Sold {qty} shares at ${fill_price:.2f}"
                        else:
                            logger.error(f"Failed to update position for {symbol} SELL_SHORT order")
                            message = f"Short sell order failed: atomic update error"
                    else:
                        message = f"Short sell order failed: {res.msg}"
                else:
                    message = f"Short signal rejected: {msg}"
            else:
                message = "Sell signal: No position to close (short selling disabled)"

        return SymbolResult(
            symbol=symbol,
            signal=signal_value,
            price=price_float,
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
                self.monitor.record_symbol_processed(
                    symbol, success=result.executed or result.signal == 0
                )
                return result

        # Process all symbols concurrently with exception safety
        tasks = [process_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and collect valid results
        valid_results = []
        market_prices = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                import traceback

                tb_str = "".join(
                    traceback.format_exception(type(result), result, result.__traceback__)
                )
                logger.error(f"Task failed for symbol {symbol}: {result}\nTraceback:\n{tb_str}")
                # Continue processing other symbols, don't crash the entire run
                continue
            else:
                valid_results.append(result)

        # Use valid_results instead of results for further processing
        for result in valid_results:
            if isinstance(result, SymbolResult):
                logger.info(
                    f"{result.symbol}: Signal={result.signal}, "
                    f"Price=${result.price:.2f}, {result.message}"
                )
                # Collect current market prices
                if result.price > 0:
                    market_prices[result.symbol] = result.price
            else:
                logger.error(f"Error processing symbol: {result}")

        # Update market prices for all positions
        await self.update_position_market_prices(market_prices)

        return [r for r in results if isinstance(r, SymbolResult)]

    async def update_position_market_prices(self, market_prices: Dict[str, float]):
        """Update market prices for all positions in database."""
        try:
            for symbol, position in self.positions.items():
                if symbol in market_prices:
                    current_price = market_prices[symbol]
                    # Update position in database with current market price
                    await self.db.update_position(
                        symbol,
                        position.quantity,
                        position.avg_price,  # Keep avg_cost same
                        current_price,  # Update market price
                    )
                    logger.debug(f"Updated {symbol} market price to ${current_price:.2f}")
        except Exception as e:
            logger.error(f"Error updating position market prices: {e}")

    async def update_account_summary(self):
        """Update account summary in database."""
        # Use actual market prices from last run
        market_prices = {}
        for symbol, pos in self.positions.items():
            # Try to get latest price from database
            latest_pos = await self.db.get_position(symbol)
            if latest_pos and latest_pos.get("market_price"):
                market_prices[symbol] = latest_pos["market_price"]
            else:
                market_prices[symbol] = float(pos.avg_price)
        equity = await self.portfolio.equity(market_prices)
        unrealized = await self.portfolio.compute_unrealized(market_prices)

        # Convert Decimal to float for APIs that expect float
        equity_float = float(equity)
        unrealized_float = float(unrealized)
        # Handle negative cash (margin) - log warning but preserve actual value
        raw_cash = float(self.portfolio.cash)
        if raw_cash < 0:
            logger.warning(f"Negative cash balance: ${raw_cash:,.2f} (margin usage)")
        cash_float = raw_cash  # Preserve actual cash value, don't clamp
        realized_pnl_float = float(self.portfolio.realized_pnl)

        # Update portfolio manager capital and consider rebalancing
        if self.portfolio_manager:
            try:
                self.portfolio_manager.update_capital(equity_float)
                if await self.portfolio_manager.should_rebalance():
                    rb = await self.portfolio_manager.rebalance()
                    logger.info(f"Rebalanced strategies at {rb['timestamp']}: {rb['new_weights']}")
            except Exception as e:
                logger.debug(f"Portfolio manager update failed: {e}")

        await self.db.update_account(
            cash=cash_float,
            equity=equity_float,
            daily_pnl=self.daily_pnl,
            realized_pnl=realized_pnl_float,
            unrealized_pnl=unrealized_float,
        )

        # Save daily equity snapshot for portfolio value tracking (industry standard)
        positions_value = sum(
            float(pos.quantity) * market_prices.get(symbol, float(pos.avg_price))
            for symbol, pos in self.positions.items()
        )
        await self.db.save_equity_snapshot(
            equity=equity_float,
            cash=cash_float,
            positions_value=positions_value,
            realized_pnl=realized_pnl_float,
            unrealized_pnl=unrealized_float,
        )

        # Save ML predictions to file for dashboard
        if self._ml_predictions:
            try:
                import json
                from pathlib import Path

                predictions_file = Path("ml_predictions.json")
                with open(predictions_file, "w") as f:
                    json.dump(self._ml_predictions, f, indent=2)
                logger.debug(
                    f"Saved {len(self._ml_predictions)} ML predictions to {predictions_file}"
                )
            except Exception as e:
                logger.warning(f"Failed to save ML predictions: {e}")

        logger.info(
            f"Trading cycle complete. Equity: ${equity:,.2f}, "
            f"Daily P&L: ${self.daily_pnl:,.2f}, "
            f"Positions: {len(self.positions)}"
        )

        # Check kill switch conditions after updating account
        if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
            kill_switch = self.advanced_risk.kill_switch
            # Use session start equity (captured after loading positions)
            starting_equity = self.session_start_equity or float(self.cfg.default_cash)
            # Guard against division by zero - use config default if starting_equity is 0 or negative
            if starting_equity <= 0:
                fallback = float(self.cfg.default_cash)
                logger.warning(
                    f"Invalid starting_equity={starting_equity}, using config default {fallback}"
                )
                starting_equity = fallback
            if kill_switch.check_daily_loss(equity_float, starting_equity):
                loss_pct = ((starting_equity - equity_float) / starting_equity) * 100
                reason = (
                    f"Daily loss {loss_pct:.1f}% exceeded limit. "
                    f"Current: ${equity_float:,.2f}, Started: ${starting_equity:,.2f}"
                )
                logger.critical(f"KILL SWITCH TRIGGERED: {reason}")
                # Trigger the kill switch and halt trading
                kill_switch.trigger(reason)
                raise KillSwitchTriggeredError(reason)

    async def run(self, symbols: Optional[List[str]] = None):
        """Main run method - process all symbols and update account."""
        await self.setup()
        try:
            # Check market status
            if not is_trading_allowed():
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

            # AI Opportunity Scanner - find new stocks to buy from news
            ai_opportunities = []
            if self.ai_analyst and AI_ANALYST_AVAILABLE and is_trading_allowed():
                try:
                    # Get current positions to exclude
                    owned_symbols = list(self.positions.keys())

                    # Fetch fresh news from diverse sources
                    news_items = fetch_rss_news(max_items=50)
                    headlines = [item["title"] for item in news_items]

                    if headlines:
                        logger.info(f"AI scanning {len(headlines)} headlines for opportunities...")
                        ai_opportunities = self.ai_analyst.find_opportunities(
                            headlines, exclude_symbols=owned_symbols
                        )

                        for opp in ai_opportunities:
                            logger.info(
                                f"🎯 AI OPPORTUNITY: {opp['symbol']} - "
                                f"Confidence: {opp['confidence']:.0%} - {opp['reason']}"
                            )
                except Exception as e:
                    logger.warning(f"AI opportunity scan failed: {e}")

            # Build symbols list - start with configured symbols
            symbols_to_process = list(symbols if symbols else self.cfg.symbols)

            # CRITICAL: Add existing positions to monitor for SELL signals
            owned_symbols = list(self.positions.keys())
            for sym in owned_symbols:
                if sym not in symbols_to_process:
                    symbols_to_process.append(sym)
            if owned_symbols:
                logger.info(
                    f"Added {len(owned_symbols)} owned positions to monitoring: {owned_symbols[:5]}..."
                )

            # Add AI-discovered opportunities to processing list
            for opp in ai_opportunities:
                if opp["symbol"] not in symbols_to_process:
                    symbols_to_process = list(symbols_to_process) + [opp["symbol"]]
                    logger.info(f"Added AI opportunity {opp['symbol']} to processing queue")

            # Store AI opportunities for signal generation
            self._ai_opportunities = {opp["symbol"]: opp for opp in ai_opportunities}
            logger.info(
                f"Processing {len(symbols_to_process)} symbols "
                f"with max {self.max_concurrent_symbols} concurrent"
            )

            # Process symbols in parallel
            results = await self.run_parallel(symbols_to_process)

            # Run pairs trading analysis if enabled AND market is open
            if (
                MEAN_REVERSION_AVAILABLE
                and self.pairs_strategy
                and len(self.market_data_cache) >= 2
                and is_trading_allowed()  # CRITICAL: Don't trade outside market hours!
            ):
                try:
                    logger.info("Running pairs trading analysis...")

                    # Find pairs if not already done
                    if (
                        not hasattr(self.pairs_strategy, "pair_stats")
                        or not self.pairs_strategy.pair_stats
                    ):
                        pairs = await self.pairs_strategy.find_pairs(
                            list(self.market_data_cache.keys()), self.market_data_cache
                        )
                        logger.info(f"Found {len(pairs)} cointegrated pairs")

                    # Analyze existing pairs
                    if self.pairs_strategy.pair_stats:
                        # Build current prices dict for ALL symbols in market_data_cache
                        current_prices = {
                            s: self.market_data_cache[s]["close"].iloc[-1]
                            for s in self.market_data_cache
                            if len(self.market_data_cache[s]) > 0
                        }

                        # Log current prices to debug
                        logger.info(
                            f"Current prices dict has {len(current_prices)} symbols: {list(current_prices.keys())[:10]}"
                        )
                        if len(current_prices) == 0:
                            logger.warning(
                                "Current prices dict is empty - market_data_cache may not be populated"
                            )

                        pairs_signals = await self.pairs_strategy.analyze_pairs(
                            list(self.pairs_strategy.pair_stats.keys()), current_prices
                        )

                        # Fetch DB positions ONCE before loop (avoid N+1 query pattern)
                        db_positions_list = await self.db.get_positions()

                        for signal in pairs_signals:
                            logger.info(
                                "pairs_signal",
                                pair=signal.get("pair"),
                                signal_type=signal.get("signal"),
                                z_score=signal.get("z_score"),
                                confidence=signal.get("confidence"),
                            )

                            # Execute pairs trades
                            if signal.get("signal") != "hold":
                                # Check if we already have a position in either symbol of the pair
                                pair = signal.get("pair")
                                signal_type = signal.get("signal")

                                if pair and len(pair) == 2:
                                    symbol_a, symbol_b = pair[0], pair[1]
                                    # Skip if we already have significant positions in these symbols
                                    # Check for existing positions (qty > 0, not > 100)
                                    has_position_a = (
                                        symbol_a in self.positions
                                        and self.positions[symbol_a].quantity > 0
                                    )
                                    has_position_b = (
                                        symbol_b in self.positions
                                        and self.positions[symbol_b].quantity > 0
                                    )

                                    if has_position_a or has_position_b:
                                        logger.info(
                                            f"Skipping pairs trade for {symbol_a}-{symbol_b}: already have positions (in-memory)"
                                        )
                                        continue

                                    # CRITICAL: Also check DB positions in case in-memory is out of sync
                                    # (uses db_positions_list fetched once before loop)
                                    db_pos_a = next(
                                        (
                                            p
                                            for p in db_positions_list
                                            if p["symbol"] == symbol_a and p.get("quantity", 0) > 0
                                        ),
                                        None,
                                    )
                                    db_pos_b = next(
                                        (
                                            p
                                            for p in db_positions_list
                                            if p["symbol"] == symbol_b and p.get("quantity", 0) > 0
                                        ),
                                        None,
                                    )
                                    if db_pos_a or db_pos_b:
                                        logger.warning(
                                            f"DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}: "
                                            f"DB position exists (A={db_pos_a is not None}, B={db_pos_b is not None})"
                                        )
                                        continue

                                    # Check MAX_OPEN_POSITIONS limit before opening pairs trades
                                    # Pairs trades open 2 positions, so check if we have room
                                    current_position_count = len(
                                        [p for p in self.positions.values() if p.quantity > 0]
                                    )
                                    max_positions = self.cfg.risk.max_open_positions
                                    if current_position_count + 2 > max_positions:
                                        logger.warning(
                                            f"POSITION LIMIT: Skipping pairs trade for {symbol_a}-{symbol_b}: "
                                            f"would exceed max positions ({current_position_count}+2 > {max_positions})"
                                        )
                                        continue

                                    # CRITICAL: Check DB for recent BUY trades to prevent duplicates
                                    # This catches trades from main strategy or previous pairs cycles
                                    # Use 600 seconds (10 min) to catch trades across multiple cycles
                                    recent_buy_a = await self.db.has_recent_buy_trade(
                                        symbol_a, seconds=600
                                    )
                                    recent_buy_b = await self.db.has_recent_buy_trade(
                                        symbol_b, seconds=600
                                    )

                                    if recent_buy_a or recent_buy_b:
                                        logger.warning(
                                            f"DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}: "
                                            f"recent BUY exists (A={recent_buy_a}, B={recent_buy_b})"
                                        )
                                        continue

                                    # Also check for recent SELL trades to prevent rapid position churn
                                    recent_sell_a = await self.db.has_recent_sell_trade(
                                        symbol_a, seconds=600
                                    )
                                    recent_sell_b = await self.db.has_recent_sell_trade(
                                        symbol_b, seconds=600
                                    )

                                    if recent_sell_a or recent_sell_b:
                                        logger.warning(
                                            f"DUPLICATE BLOCKED: Skipping pairs trade for {symbol_a}-{symbol_b}: "
                                            f"recent SELL exists (A={recent_sell_a}, B={recent_sell_b})"
                                        )
                                        continue

                                    # Update internal position tracking
                                    self.pairs_strategy.update_position(pair, signal)

                                    # Get current prices - fallback to market_data_cache if not in current_prices
                                    price_a = current_prices.get(symbol_a, 0)
                                    price_b = current_prices.get(symbol_b, 0)

                                    # If prices not found, try to get from market_data_cache directly
                                    if (
                                        price_a == 0
                                        and symbol_a in self.market_data_cache
                                        and len(self.market_data_cache[symbol_a]) > 0
                                    ):
                                        price_a = self.market_data_cache[symbol_a]["close"].iloc[-1]
                                    if (
                                        price_b == 0
                                        and symbol_b in self.market_data_cache
                                        and len(self.market_data_cache[symbol_b]) > 0
                                    ):
                                        price_b = self.market_data_cache[symbol_b]["close"].iloc[-1]

                                    logger.info(
                                        f"Pairs trade setup: {symbol_a}=${price_a:.2f}, {symbol_b}=${price_b:.2f}"
                                    )

                                    if price_a > 0 and price_b > 0:
                                        # Calculate position sizes (simplified - equal dollar amounts)
                                        # Calculate total portfolio value from positions
                                        equity = await self.portfolio.equity(current_prices)
                                        equity_float = float(equity)  # Convert Decimal to float
                                        pair_allocation = min(
                                            10000.0, equity_float * 0.02
                                        )  # Max 2% per leg
                                        qty_a = int(pair_allocation / price_a)
                                        qty_b = int(pair_allocation / price_b)

                                        # Execute based on signal type
                                        if (
                                            signal_type == "long_a_short_b"
                                            and qty_a > 0
                                            and qty_b > 0
                                        ):
                                            # Buy symbol_a
                                            order_a = Order(
                                                symbol=symbol_a,
                                                quantity=qty_a,
                                                side="BUY",
                                                price=price_a,
                                            )
                                            res_a = await self._place_order_with_circuit_breaker(
                                                order_a
                                            )
                                            if res_a.ok:
                                                fill_a = res_a.fill_price or price_a
                                                await self.db.record_trade(
                                                    symbol_a,
                                                    "BUY",
                                                    qty_a,
                                                    fill_a,
                                                    0,
                                                )
                                                # CRITICAL: Update positions to prevent duplicate buys
                                                self.positions[symbol_a] = Position(
                                                    symbol_a, qty_a, fill_a
                                                )
                                                await self.db.update_position(
                                                    symbol_a, qty_a, fill_a, fill_a
                                                )
                                                # Sync to Portfolio for equity calculation
                                                await self.portfolio.update_fill(
                                                    symbol_a, "BUY", qty_a, fill_a
                                                )
                                                logger.info(
                                                    f"Pairs trade: Bought {qty_a} {symbol_a} at ${fill_a:.2f}"
                                                )
                                                # Track trade count
                                                if hasattr(self, "trades_executed"):
                                                    self.trades_executed += 1
                                                # Add stop-loss for pairs position
                                                if self.stop_loss_monitor and self.enable_stop_loss:
                                                    try:
                                                        new_pos = Position(
                                                            symbol=symbol_a,
                                                            quantity=qty_a,
                                                            avg_price=fill_a,
                                                            entry_time=datetime.now(),
                                                        )
                                                        if self.use_trailing_stop:
                                                            await self.stop_loss_monitor.add_stop_loss(
                                                                symbol=symbol_a,
                                                                position=new_pos,
                                                                stop_percent=self.trailing_stop_pct,
                                                                stop_type=StopType.TRAILING_PERCENT,
                                                                trailing_percent=self.trailing_stop_pct,
                                                            )
                                                            logger.info(
                                                                f"Trailing stop added for pairs position {symbol_a}"
                                                            )
                                                        else:
                                                            await self.stop_loss_monitor.add_stop_loss(
                                                                symbol=symbol_a,
                                                                position=new_pos,
                                                                stop_percent=self.stop_loss_percent,
                                                                stop_type=StopType.FIXED,
                                                            )
                                                            logger.info(
                                                                f"Fixed stop-loss added for pairs position {symbol_a}"
                                                            )
                                                    except Exception as e:
                                                        logger.error(
                                                            f"Failed to add stop-loss for pairs {symbol_a}: {e}"
                                                        )

                                            # Short symbol_b (if shorting enabled, otherwise skip)
                                            if self.cfg.execution.enable_short_selling:
                                                order_b = Order(
                                                    symbol=symbol_b,
                                                    quantity=qty_b,
                                                    side="SELL",
                                                    price=price_b,
                                                )
                                                res_b = (
                                                    await self._place_order_with_circuit_breaker(
                                                        order_b
                                                    )
                                                )
                                                if res_b.ok:
                                                    await self.db.record_trade(
                                                        symbol_b,
                                                        "SELL",
                                                        qty_b,
                                                        res_b.fill_price or price_b,
                                                        0,
                                                    )
                                                    logger.info(
                                                        f"Pairs trade: Shorted {qty_b} {symbol_b} at ${price_b:.2f}"
                                                    )

                                        elif (
                                            signal_type == "long_b_short_a"
                                            and qty_a > 0
                                            and qty_b > 0
                                        ):
                                            # Buy symbol_b
                                            order_b = Order(
                                                symbol=symbol_b,
                                                quantity=qty_b,
                                                side="BUY",
                                                price=price_b,
                                            )
                                            res_b = await self._place_order_with_circuit_breaker(
                                                order_b
                                            )
                                            if res_b.ok:
                                                fill_b = res_b.fill_price or price_b
                                                await self.db.record_trade(
                                                    symbol_b,
                                                    "BUY",
                                                    qty_b,
                                                    fill_b,
                                                    0,
                                                )
                                                # CRITICAL: Update positions to prevent duplicate buys
                                                self.positions[symbol_b] = Position(
                                                    symbol_b, qty_b, fill_b
                                                )
                                                await self.db.update_position(
                                                    symbol_b, qty_b, fill_b, fill_b
                                                )
                                                # Sync to Portfolio for equity calculation
                                                await self.portfolio.update_fill(
                                                    symbol_b, "BUY", qty_b, fill_b
                                                )
                                                logger.info(
                                                    f"Pairs trade: Bought {qty_b} {symbol_b} at ${fill_b:.2f}"
                                                )
                                                if hasattr(self, "trades_executed"):
                                                    self.trades_executed += 1
                                                # Add stop-loss for pairs position
                                                if self.stop_loss_monitor and self.enable_stop_loss:
                                                    try:
                                                        new_pos = Position(
                                                            symbol=symbol_b,
                                                            quantity=qty_b,
                                                            avg_price=fill_b,
                                                            entry_time=datetime.now(),
                                                        )
                                                        if self.use_trailing_stop:
                                                            await self.stop_loss_monitor.add_stop_loss(
                                                                symbol=symbol_b,
                                                                position=new_pos,
                                                                stop_percent=self.trailing_stop_pct,
                                                                stop_type=StopType.TRAILING_PERCENT,
                                                                trailing_percent=self.trailing_stop_pct,
                                                            )
                                                            logger.info(
                                                                f"Trailing stop added for pairs position {symbol_b}"
                                                            )
                                                        else:
                                                            await self.stop_loss_monitor.add_stop_loss(
                                                                symbol=symbol_b,
                                                                position=new_pos,
                                                                stop_percent=self.stop_loss_percent,
                                                                stop_type=StopType.FIXED,
                                                            )
                                                            logger.info(
                                                                f"Fixed stop-loss added for pairs position {symbol_b}"
                                                            )
                                                    except Exception as e:
                                                        logger.error(
                                                            f"Failed to add stop-loss for pairs {symbol_b}: {e}"
                                                        )

                                            # Short symbol_a (if shorting enabled, otherwise skip)
                                            if self.cfg.execution.enable_short_selling:
                                                order_a = Order(
                                                    symbol=symbol_a,
                                                    quantity=qty_a,
                                                    side="SELL",
                                                    price=price_a,
                                                )
                                                res_a = (
                                                    await self._place_order_with_circuit_breaker(
                                                        order_a
                                                    )
                                                )
                                                if res_a.ok:
                                                    await self.db.record_trade(
                                                        symbol_a,
                                                        "SELL",
                                                        qty_a,
                                                        res_a.fill_price or price_a,
                                                        0,
                                                    )
                                                    logger.info(
                                                        f"Pairs trade: Shorted {qty_a} {symbol_a} at ${price_a:.2f}"
                                                    )

                except Exception as e:
                    logger.error(f"Pairs trading analysis error: {e}")

            # Run statistical arbitrage analysis if enabled
            if (
                MEAN_REVERSION_AVAILABLE
                and self.stat_arb_strategy
                and len(self.market_data_cache) >= 5
            ):
                try:
                    logger.info("Running statistical arbitrage analysis...")

                    arb_scores = await self.stat_arb_strategy.calculate_arbitrage_scores(
                        list(self.market_data_cache.keys()), self.market_data_cache
                    )

                    if arb_scores:
                        weights = self.stat_arb_strategy.generate_portfolio_weights(arb_scores)
                        logger.info(
                            "stat_arb_portfolio",
                            num_opportunities=len(arb_scores),
                            selected_symbols=list(weights.keys()),
                            weights={k: f"{v:.2%}" for k, v in weights.items()},
                        )

                except Exception as e:
                    logger.error(f"Statistical arbitrage analysis error: {e}")

            # Update account summary
            await self.update_account_summary()

            # Log execution summary
            executed_count = sum(1 for r in results if r.executed)
            logger.info(f"Processed {len(results)} symbols, " f"executed {executed_count} trades")

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

    async def cancel_all_orders(self):
        """Cancel all pending orders - used for emergency shutdown."""
        try:
            cancelled_count = 0

            # Cancel stop-loss orders
            if self.stop_loss_monitor:
                cancelled_count += self.stop_loss_monitor.cancel_all_stops()

            # In a real system, would also cancel any pending broker orders
            # For paper trading, we just log the action
            logger.warning(f"Emergency shutdown: Cancelled {cancelled_count} orders")

            # Clear any pending executions
            self.positions.clear()

            return cancelled_count
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return 0

    async def cleanup(self):
        """Clean up resources when runner is done."""
        try:
            # Cancel subprocess monitor task if running
            if self.subprocess_monitor_task and not self.subprocess_monitor_task.done():
                self.subprocess_monitor_task.cancel()
                try:
                    await self.subprocess_monitor_task
                except asyncio.CancelledError:
                    pass

            # Cancel risk monitor task if running
            if self.risk_monitor_task and not self.risk_monitor_task.done():
                self.risk_monitor_task.cancel()
                try:
                    await self.risk_monitor_task
                except asyncio.CancelledError:
                    pass

            # Stop and cleanup stop-loss monitor
            if self.stop_loss_monitor:
                logger.info("Stopping stop-loss monitor...")
                await self.stop_loss_monitor.stop_monitoring()
                metrics = self.stop_loss_monitor.get_metrics()
                logger.info(
                    f"Stop-loss metrics - Triggered: {metrics.triggered_today}, "
                    f"Executed: {metrics.executed_today}, Failed: {metrics.failed_today}, "
                    f"Prevented loss: ${metrics.total_prevented_loss:.2f}"
                )

            # Save advanced risk manager state
            if self.use_advanced_risk and self.advanced_risk:
                state_file = Path("data/risk_state.json")
                state_file.parent.mkdir(exist_ok=True)
                self.advanced_risk.save_state(state_file)
                logger.info("Advanced risk manager state saved")

            # Disconnect from IBKR (subprocess client)
            if hasattr(self, "ib") and self.ib:
                logger.info("Disconnecting from IBKR...")
                # Check if it's subprocess client or legacy IB client
                if hasattr(self.ib, "disconnect"):
                    # Subprocess client has async disconnect()
                    if asyncio.iscoroutinefunction(self.ib.disconnect):
                        await self.ib.disconnect()
                    else:
                        self.ib.disconnect()
                # Also stop the subprocess if it exists
                if hasattr(self.ib, "stop"):
                    await self.ib.stop()
                    logger.info("Stopped IBKR subprocess")

            # Close database connections
            if hasattr(self, "db") and self.db:
                await self.db.close()

            # Stop WebSocket updates
            if hasattr(self, "ws_client"):
                self.ws_client.stop()

            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def run_once(
    symbols: Optional[List[str]] = None,
    duration: str = "1 D",
    bar_size: str = "1 min",
    sma_fast: int = 10,
    sma_slow: int = 20,
    slippage_bps: float = 0.0,
    max_order_notional: Optional[float] = None,
    max_daily_notional: Optional[float] = None,
    default_cash: Optional[float] = None,
    max_concurrent: int = 8,
    use_ml_strategy: bool = False,
    use_ml_enhanced: bool = False,
    use_smart_execution: bool = False,
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
        use_ml_strategy=use_ml_strategy,
        use_ml_enhanced=use_ml_enhanced,
        use_smart_execution=use_smart_execution,
    )
    await runner.run(symbols)


async def run_continuous(
    symbols: Optional[List[str]] = None,
    duration: str = "1 D",
    bar_size: str = "1 min",
    sma_fast: int = 10,
    sma_slow: int = 20,
    slippage_bps: float = 0.0,
    max_order_notional: Optional[float] = None,
    max_daily_notional: Optional[float] = None,
    default_cash: Optional[float] = None,
    max_concurrent: int = 8,
    interval_seconds: int = 15,  # 15 seconds between cycles for near real-time
    use_ml_strategy: bool = False,
    use_ml_enhanced: bool = False,
    use_smart_execution: bool = False,
    force_connect: bool = False,
) -> None:
    """Run the trading system continuously with market hours checking."""
    import signal
    from datetime import datetime

    import pytz

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

    # Persistent runner - created once, reused across cycles
    runner = None

    try:
        while not shutdown_flag:
            eastern = pytz.timezone("US/Eastern")
            current_time = datetime.now(eastern)

            # Check market status and adjust polling frequency
            if not is_trading_allowed() and not force_connect:
                session = get_market_session()
                seconds_to_open = seconds_until_market_open()

                # Extended hours: slower polling (2 min) - but only if extended hours trading is DISABLED
                # If ENABLE_EXTENDED_HOURS=true, is_trading_allowed() returns true and we won't reach here
                if session in ["after-hours", "pre-market"]:
                    wait_time = 120  # 2 minutes during extended hours
                    logger.info(f"Extended hours ({session}). Polling every 2 minutes...")
                    await asyncio.sleep(wait_time)
                    continue
                # Within 1 hour of open: moderate polling (5 min)
                elif seconds_to_open < 3600:
                    wait_time = 300  # 5 minutes when close to open
                    logger.info(
                        f"Market opens in {seconds_to_open/60:.0f} min. Polling every 5 minutes..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # Market fully closed (overnight/weekend): long wait (30 min max)
                    wait_time = min(1800, seconds_to_open // 2)  # Max 30 minutes
                    logger.info(
                        f"Market {session}. Next open in {seconds_to_open/3600:.1f} hours. "
                        f"Sleeping {wait_time/60:.1f} minutes..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
            elif force_connect and not is_trading_allowed():
                logger.warning(
                    "⚠️ Force-connect enabled - connecting to IBKR despite market being closed"
                )

            try:
                logger.info(
                    f"Starting trading cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )

                # Load portfolio configurations
                from .multiuser.portfolio_config import PortfolioConfig, load_portfolio_configs

                try:
                    portfolio_configs = load_portfolio_configs()
                except Exception as pc_err:
                    logger.warning(f"Failed to load portfolio configs: {pc_err}, using default")
                    portfolio_configs = [
                        PortfolioConfig(
                            id="default",
                            name="Default Portfolio",
                            starting_cash=default_cash or 100000,
                            symbols=symbols or [],
                        )
                    ]

                active_portfolios = [pc for pc in portfolio_configs if pc.active]
                logger.info(
                    f"Processing {len(active_portfolios)} active portfolio(s): "
                    f"{[p.id for p in active_portfolios]}"
                )

                for portfolio_cfg in active_portfolios:
                    # Determine symbols for this portfolio
                    portfolio_symbols = portfolio_cfg.symbols if portfolio_cfg.symbols else symbols

                    logger.info(
                        f"── Portfolio '{portfolio_cfg.id}' ({portfolio_cfg.name}): "
                        f"{len(portfolio_symbols or [])} symbols ──"
                    )

                    # Create fresh runner each cycle for stability (disconnect between cycles)
                    # This avoids connection timeouts and subprocess health check issues
                    runner = AsyncRunner(
                        duration=duration,
                        bar_size=bar_size,
                        sma_fast=sma_fast,
                        sma_slow=sma_slow,
                        slippage_bps=slippage_bps,
                        max_order_notional=max_order_notional,
                        max_daily_notional=max_daily_notional,
                        default_cash=portfolio_cfg.starting_cash,
                        max_concurrent_symbols=max_concurrent,
                        use_correlation_sizing=True,  # FIXED: Enabled M5 correlation integration
                        use_ml_strategy=use_ml_strategy,
                        use_smart_execution=use_smart_execution,
                        portfolio_id=portfolio_cfg.id,
                    )

                    await runner.run(portfolio_symbols)

                    # Clean up connection after each portfolio (more stable for long-running)
                    logger.info(f"Cleaning up runner for portfolio '{portfolio_cfg.id}'...")
                    await runner.cleanup()
                    runner = None

                # Wait before next iteration
                if not shutdown_flag and is_trading_allowed():
                    logger.info(
                        f"Waiting {interval_seconds/60:.1f} minutes before next iteration..."
                    )
                    await asyncio.sleep(interval_seconds)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except KillSwitchTriggeredError as e:
                # Kill switch triggered - graceful shutdown
                logger.critical(f"KILL SWITCH: Shutting down trading system - {e}")
                shutdown_flag = True
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                # If connection error, reset runner to force reconnect
                if "connect" in str(e).lower() or "timeout" in str(e).lower():
                    logger.warning("Connection error detected, will reconnect on next cycle")
                    if runner:
                        await runner.cleanup()
                    runner = None
                if not shutdown_flag:
                    logger.info("Waiting 1 minute before retry...")
                    await asyncio.sleep(60)

    finally:
        # Cleanup on shutdown
        if runner:
            logger.info("Cleaning up persistent runner...")
            await runner.cleanup()
        logger.info("Trading system shutdown complete")


def check_gateway_zombies(port: int = 4002) -> bool:
    """
    Check for zombie CLOSE_WAIT connections before starting.

    Args:
        port: IBKR Gateway port to check (default: 4002 for paper)

    Returns:
        True if no zombies detected, False if zombies found
    """
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:CLOSE_WAIT"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.stdout.strip():
            logger.error(f"Detected zombie CLOSE_WAIT connections on port {port}")
            logger.error(result.stdout)
            logger.error("Gateway may be full. Solutions:")
            logger.error("  1. Restart Gateway (File → Exit → Restart with 2FA)")
            logger.error("  2. Or try: ./START_TRADER.sh (has automatic cleanup)")
            return False

        logger.info(f"✓ No zombie connections detected on port {port}")
        return True

    except subprocess.TimeoutExpired:
        logger.warning("lsof command timed out - skipping zombie check")
        return True  # Don't block startup
    except FileNotFoundError:
        logger.debug("lsof not available - skipping zombie check")
        return True  # Don't block startup on systems without lsof
    except Exception as e:
        logger.warning(f"Error checking for zombie connections: {e}")
        return True  # Don't block startup on unexpected errors


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
        "--duration", type=str, help="IB duration string (e.g. '1 D')", default="1 D"
    )
    parser.add_argument("--bar-size", type=str, help="IB bar size (e.g. '1 min')", default="1 min")
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
        default=15,
        help="Seconds between trading cycles (default: 15 for near real-time)",
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        help="Use ML strategy instead of SMA crossover",
    )
    parser.add_argument(
        "--use-ml-enhanced",
        action="store_true",
        help="Use ML Enhanced strategy with regime detection and multi-timeframe analysis",
    )
    parser.add_argument(
        "--use-smart-execution",
        action="store_true",
        help="Enable smart execution algorithms (TWAP, VWAP, Iceberg)",
    )
    parser.add_argument(
        "--force-connect",
        action="store_true",
        help="Force IBKR connection even when market is closed (for testing)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if cfg.execution.mode == "live" and not args.confirm_live:
        raise SystemExit("Refusing to run in live mode without --confirm-live")

    # Check for zombie connections before starting
    # Gateway-owned zombies will be handled in setup() by triggering Gateway restart
    port = cfg.ibkr.port  # Get port from config (4002 for paper, 4001 for live)
    if not check_gateway_zombies(port):
        logger.warning("Zombie connections detected - setup() will attempt cleanup/restart")

    override_symbols = (
        [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
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
                use_ml_strategy=args.use_ml,
                use_ml_enhanced=args.use_ml_enhanced,
                use_smart_execution=args.use_smart_execution,
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
                use_ml_strategy=args.use_ml,
                use_ml_enhanced=args.use_ml_enhanced,
                use_smart_execution=args.use_smart_execution,
                force_connect=args.force_connect,
            )
        )


if __name__ == "__main__":
    main()
