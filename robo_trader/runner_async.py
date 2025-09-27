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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .analysis.correlation_integration import AsyncCorrelationManager, CorrelationBasedPositionSizer
from .config import load_config
from .connection_manager import ConnectionManager
from .correlation import CorrelationTracker
from .database_async import AsyncTradingDatabase
from .execution import Order, PaperExecutor
from .logger import get_logger
from .market_hours import get_market_session, is_market_open, seconds_until_market_open
from .monitoring.performance import PerformanceMonitor, Timer

# Import WebSocket client for real-time updates
try:
    from .websocket_client import ws_client

    WEBSOCKET_ENABLED = True
except ImportError:
    ws_client = None
    WEBSOCKET_ENABLED = False
from .portfolio import Portfolio  # Import Portfolio class from portfolio.py file
from .portfolio_pkg.portfolio_manager import AllocationMethod, MultiStrategyPortfolioManager
from .risk.advanced_risk import AdvancedRiskManager, risk_monitor_task
from .risk_manager import Position, RiskManager
from .stop_loss_monitor import StopLossMonitor, StopType
from .strategies import MLStrategy, sma_crossover_signals
from .strategies.ml_enhanced_strategy import MLEnhancedStrategy

# Import mean reversion strategies if available
logger = get_logger(__name__)

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
        use_ml_strategy: bool = False,
        use_ml_enhanced: bool = None,  # Auto-detect if None
        use_smart_execution: bool = None,  # Auto-detect if None
        use_advanced_risk: bool = None,  # Auto-detect if None
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
        self.conn_mgr = None
        self.ib = None
        self.db = None
        self.risk = None
        self.advanced_risk = None  # Advanced risk manager with Kelly sizing
        self.risk_monitor_task = None  # Background risk monitoring task
        self.executor = None
        self.portfolio = None
        self.portfolio_manager: Optional[MultiStrategyPortfolioManager] = None
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_executed_notional = 0.0
        self.monitor = PerformanceMonitor()

        # Position locks to prevent race conditions
        self._position_locks: Dict[str, asyncio.Lock] = {}
        self._position_lock_manager = asyncio.Lock()

        # Correlation components
        self.correlation_tracker = None
        self.position_sizer = None
        self.correlation_manager = None

        # ML Strategy
        self.ml_strategy = None

        # Stop-loss monitoring
        self.stop_loss_monitor = None
        self.enable_stop_loss = True  # Always enabled for safety
        self.stop_loss_percent = 0.02  # Default 2% stop-loss
        self.ml_enhanced_strategy = None
        self.active_strategy_name: Optional[str] = None

        # Mean reversion strategies
        self.mean_reversion_strategy = None
        self.pairs_strategy = None
        self.stat_arb_strategy = None
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
                                self.positions[symbol] = Position(symbol, new_qty, pos.avg_price)
                        elif side.upper() == "BUY" and pos.quantity >= 0:
                            # Adding to long position
                            total_qty = pos.quantity + quantity
                            new_avg = (pos.avg_price * pos.quantity + price * quantity) / total_qty
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
                                self.positions[symbol] = Position(symbol, remaining, pos.avg_price)
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

                # Update portfolio
                self.portfolio.update_fill(symbol, side, quantity, price)

                # Update advanced risk manager if enabled
                if self.use_advanced_risk and self.advanced_risk:
                    self.advanced_risk.update_position(symbol, quantity, price, side)

                return True

            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {e}")
                return False

    async def setup(self):
        """Initialize all components."""
        self.cfg = load_config()

        # CRITICAL PRE-FLIGHT CHECK: Test IBKR connection before proceeding
        # This prevents the system from starting without a working connection
        logger.info("=" * 60)
        logger.info("PERFORMING IBKR PRE-FLIGHT CONNECTION CHECK")
        logger.info("=" * 60)

        # Test if configured TWS/IB Gateway port is open
        import socket
        import sys

        def test_port_open(host="127.0.0.1", port=7497, timeout=5):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((host, port))
                sock.close()
                return result == 0
            except Exception as e:
                logger.error(f"Port test failed: {e}")
                return False

        host = self.cfg.ibkr.host
        port = self.cfg.ibkr.port

        if not test_port_open(host=host, port=port):
            logger.error(f"❌ IBKR PRE-FLIGHT CHECK FAILED")
            logger.error(f"Port {port} is not open on host {host} - TWS/IB Gateway not running")
            logger.error("Please ensure TWS or IB Gateway is running and configured properly:")
            logger.error("1. Start TWS/IB Gateway")
            logger.error("2. Enable API connections in Global Configuration")
            logger.error("3. Set correct port (7497 for paper, 7496 for live)")
            logger.error("4. Allow connections from localhost")
            logger.error("REFUSING TO START WITHOUT IBKR CONNECTION")
            sys.exit(1)

        logger.info(f"✓ Port {port} is open - proceeding to IBKR connect")

        # Initialize robust connection manager (uses env-configured client ID with rotation on conflict)
        self.conn_mgr = ConnectionManager(host=host, port=port, client_id=self.cfg.ibkr.client_id)

        # Try to connect with timeout and verify
        try:
            self.ib = await asyncio.wait_for(self.conn_mgr.connect(), timeout=30)
            logger.info("✓ IBKR connection established successfully")
            logger.info("=" * 60)
        except asyncio.TimeoutError:
            logger.error("❌ IBKR CONNECTION TIMEOUT")
            logger.error("Failed to connect to IBKR after 30 seconds")
            logger.error("Check TWS API configuration and try again")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ IBKR CONNECTION FAILED: {e}")
            logger.error("Cannot proceed without IBKR connection")
            sys.exit(1)

        # Initialize async database with context manager
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

        # Initialize advanced risk manager with Kelly sizing and kill switches
        if self.use_advanced_risk:
            logger.info("Initializing advanced risk management with Kelly sizing and kill switches")
            self.advanced_risk = AdvancedRiskManager(
                config={
                    "starting_capital": self.default_cash
                    if self.default_cash is not None
                    else self.cfg.default_cash,
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
            )
            await self.stop_loss_monitor.start_monitoring()
            logger.info(f"Stop-loss monitoring enabled with {self.stop_loss_percent:.1%} threshold")

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

        # Load existing positions from database to prevent duplicate buying
        await self.load_existing_positions()

        logger.info("AsyncRunner setup complete")

    async def load_existing_positions(self):
        """Load existing positions from database on startup to prevent duplicate buying."""
        try:
            positions_data = await self.db.get_positions()
            for pos in positions_data:
                if pos.get("quantity", 0) > 0:  # Only load open positions
                    symbol = pos["symbol"]
                    quantity = pos["quantity"]
                    avg_cost = pos.get("avg_cost", pos.get("price", 0))

                    # Create Position object and add to positions dict
                    self.positions[symbol] = Position(symbol, quantity, avg_cost)
                    logger.info(
                        f"Loaded existing position: {symbol} qty={quantity} avg_cost=${avg_cost:.2f}"
                    )

            if self.positions:
                logger.info(
                    f"Loaded {len(self.positions)} existing positions from database: {list(self.positions.keys())}"
                )
            else:
                logger.info("No existing positions found in database")

        except Exception as e:
            logger.error(f"Failed to load existing positions from database: {e}")
            logger.warning("Starting with empty positions - may result in duplicate trades!")

    async def teardown(self):
        """Clean up resources."""
        if self.correlation_manager:
            await self.correlation_manager.stop()
        if self.conn_mgr:
            await self.conn_mgr.disconnect()
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
                # Use ConnectionManager to fetch historical bars
                df = await self.conn_mgr.fetch_historical_bars(
                    symbol=symbol, duration=self.duration, bar_size=self.bar_size
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
                await self.monitor.record_data_points(len(batch_data))

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _manage_cache_size(self):
        """Ensure cache doesn't exceed max size (LRU eviction)."""
        while len(self.market_data_cache) > self.max_cache_size:
            # Remove oldest item (first item in OrderedDict)
            oldest_symbol = next(iter(self.market_data_cache))
            del self.market_data_cache[oldest_symbol]
            logger.debug(f"Evicted {oldest_symbol} from market data cache")

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

        # Cache market data for pairs/stat arb analysis with LRU eviction
        if symbol in self.market_data_cache:
            self.market_data_cache.move_to_end(symbol)
        self.market_data_cache[symbol] = df
        self._manage_cache_size()

        # Send real-time price update via WebSocket
        latest_price = None
        if df is not None and not df.empty:
            latest_price = float(df["close"].iloc[-1])

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
                else:
                    signal_value = 0
                    confidence = 0.5
                    position_size = 0.02
                    stop_loss = 0.02
                    take_profit = 0.05

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
        price = float(last.get("close", df["close"].iloc[-1]))

        # Get current equity
        equity = self.portfolio.equity({symbol: price for symbol in self.positions})

        # Record signal in database
        signal_value = int(last.get("signal", 0))
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
                        message = f"Kill switch active - trading halted"
                        return SymbolResult(
                            symbol=symbol,
                            signal=0,  # No signal computed when kill switch active
                            price=price,
                            quantity=0,
                            executed=False,
                            message=message,
                            data=df,
                        )

                with Timer("order_execution", self.monitor):
                    res = self.executor.place_order(
                        Order(
                            symbol=symbol, quantity=qty_to_cover, side="BUY_TO_COVER", price=price
                        )
                    )
                if res.ok:
                    # Cancel stop-loss when covering short position
                    if self.stop_loss_monitor:
                        self.stop_loss_monitor.cancel_stop(symbol)
                    fill_price = res.fill_price or price
                    # Use atomic position update to prevent race conditions
                    success = await self._update_position_atomic(
                        symbol, qty_to_cover, fill_price, "BUY_TO_COVER"
                    )
                    if success:
                        self.daily_pnl = self.portfolio.realized_pnl

                        # Record trade in database
                        await self.db.record_trade(
                            symbol,
                            "BUY_TO_COVER",
                            qty_to_cover,
                            fill_price,
                            slippage=(fill_price - price) * qty_to_cover if res.fill_price else 0,
                        )
                        await self.db.update_position(symbol, 0, 0, 0)  # Close position

                        await self.monitor.record_order_placed(symbol, qty_to_cover)
                        await self.monitor.record_trade_executed(
                            symbol, "BUY_TO_COVER", qty_to_cover
                        )
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
                # Open long position
                # Use advanced risk manager with Kelly sizing if enabled
                if self.use_advanced_risk and self.advanced_risk:
                    # Get ATR for stop loss calculation (if available)
                    atr = df["atr"].iloc[-1] if "atr" in df.columns else None

                    # Calculate position size using Kelly criterion
                    sizing_result = await self.advanced_risk.calculate_position_size(
                        symbol=symbol,
                        signal_strength=abs(signal_value),  # Use signal strength from strategy
                        current_price=price,
                        atr=atr,
                    )

                    if sizing_result["blocked"]:
                        logger.warning(
                            f"Trade blocked by kill switch for {symbol}: {sizing_result['block_reason']}"
                        )
                        executed = False
                        message = f"Blocked: {sizing_result['block_reason']}"
                        return SymbolResult(symbol, signal_value, price, 0, executed, message, df)

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
                            message = f"Kill switch active - trading halted"
                            return SymbolResult(
                                symbol=symbol,
                                signal=0,  # No signal computed when kill switch active
                                price=price,
                                quantity=0,
                                executed=False,
                                message=message,
                                data=df,
                            )

                    with Timer("order_execution", self.monitor):
                        res = self.executor.place_order(
                            Order(symbol=symbol, quantity=qty, side="BUY", price=price)
                        )
                    if res.ok:
                        fill_price = res.fill_price or price
                        # Use atomic position update to prevent race conditions
                        success = await self._update_position_atomic(symbol, qty, fill_price, "BUY")
                        if success:
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
                                    await self.stop_loss_monitor.add_stop_loss(
                                        symbol=symbol,
                                        position=new_position,
                                        stop_percent=self.stop_loss_percent,
                                        stop_type=StopType.FIXED,
                                    )
                                    logger.info(
                                        f"Stop-loss order added for {symbol} at {self.stop_loss_percent:.1%}"
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
            else:
                message = "Buy signal: Already have long position"

        elif signal_value == -1:  # Sell signal
            enable_short_selling = self.cfg.execution.enable_short_selling

            if symbol in self.positions:
                # Close long position or cover short
                pos = self.positions[symbol]

                if pos.quantity > 0:  # Closing long position
                    # Check kill switch before order execution
                    if self.advanced_risk and hasattr(self.advanced_risk, "kill_switch"):
                        if self.advanced_risk.kill_switch.triggered:
                            logger.error(f"KILL SWITCH ACTIVE - Order blocked for {symbol} SELL")
                            message = f"Kill switch active - trading halted"
                            return SymbolResult(
                                symbol=symbol,
                                signal=0,  # No signal computed when kill switch active
                                price=price,
                                quantity=0,
                                executed=False,
                                message=message,
                                data=df,
                            )

                    with Timer("order_execution", self.monitor):
                        res = self.executor.place_order(
                            Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=price)
                        )
                    if res.ok:
                        # Cancel stop-loss when closing position
                        if self.stop_loss_monitor:
                            self.stop_loss_monitor.cancel_stop(symbol)
                        fill_price = res.fill_price or price
                        # Use atomic position update to prevent race conditions
                        success = await self._update_position_atomic(
                            symbol, pos.quantity, fill_price, "SELL"
                        )
                        if success:
                            self.daily_pnl = self.portfolio.realized_pnl

                            # Record trade in database
                            await self.db.record_trade(
                                symbol,
                                "SELL",
                                pos.quantity,
                                fill_price,
                                slippage=(
                                    (fill_price - price) * pos.quantity if res.fill_price else 0
                                ),
                            )
                            await self.db.update_position(symbol, 0, 0, 0)  # Close position

                            await self.monitor.record_order_placed(symbol, pos.quantity)
                            await self.monitor.record_trade_executed(symbol, "SELL", pos.quantity)
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
                            message = f"Kill switch active - trading halted"
                            return SymbolResult(
                                symbol=symbol,
                                signal=0,  # No signal computed when kill switch active
                                price=price,
                                quantity=0,
                                executed=False,
                                message=message,
                                data=df,
                            )

                    with Timer("order_execution", self.monitor):
                        res = self.executor.place_order(
                            Order(symbol=symbol, quantity=qty, side="SELL_SHORT", price=price)
                        )
                    if res.ok:
                        fill_price = res.fill_price or price
                        # Add stop-loss for short position
                        if self.stop_loss_monitor and self.enable_stop_loss:
                            try:
                                # Create position object for stop-loss (negative quantity for short)
                                short_position = Position(
                                    symbol=symbol,
                                    quantity=-qty,  # Negative for short
                                    avg_price=fill_price,
                                    entry_time=datetime.now(),
                                )
                                await self.stop_loss_monitor.add_stop_loss(
                                    symbol=symbol,
                                    position=short_position,
                                    stop_percent=self.stop_loss_percent,
                                    stop_type=StopType.FIXED,
                                )
                                logger.info(
                                    f"Stop-loss order added for SHORT {symbol} at {self.stop_loss_percent:.1%}"
                                )
                            except Exception as e:
                                logger.error(f"Failed to add stop-loss for short {symbol}: {e}")
                        fill_price = res.fill_price or price
                        # Use atomic position update to prevent race conditions
                        success = await self._update_position_atomic(
                            symbol, qty, fill_price, "SELL_SHORT"
                        )
                        if success:
                            self.daily_executed_notional += price * qty

                            # Record trade in database
                            await self.db.record_trade(
                                symbol,
                                "SELL_SHORT",
                                qty,
                                fill_price,
                                slippage=(fill_price - price) * qty if res.fill_price else 0,
                            )
                            await self.db.update_position(symbol, -qty, fill_price, price)

                            await self.monitor.record_order_placed(symbol, qty)
                            await self.monitor.record_trade_executed(symbol, "SELL_SHORT", qty)
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

        # Log results and collect market prices
        market_prices = {}
        for result in results:
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
                market_prices[symbol] = pos.avg_price
        equity = self.portfolio.equity(market_prices)
        unrealized = self.portfolio.compute_unrealized(market_prices)

        # Update portfolio manager capital and consider rebalancing
        if self.portfolio_manager:
            try:
                self.portfolio_manager.update_capital(equity)
                if await self.portfolio_manager.should_rebalance():
                    rb = await self.portfolio_manager.rebalance()
                    logger.info(f"Rebalanced strategies at {rb['timestamp']}: {rb['new_weights']}")
            except Exception as e:
                logger.debug(f"Portfolio manager update failed: {e}")

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

            # Run pairs trading analysis if enabled
            if (
                MEAN_REVERSION_AVAILABLE
                and self.pairs_strategy
                and len(self.market_data_cache) >= 2
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
                                    has_position_a = (
                                        symbol_a in self.positions
                                        and self.positions[symbol_a].quantity > 100
                                    )
                                    has_position_b = (
                                        symbol_b in self.positions
                                        and self.positions[symbol_b].quantity > 100
                                    )

                                    if has_position_a or has_position_b:
                                        logger.info(
                                            f"Skipping pairs trade for {symbol_a}-{symbol_b}: already have positions"
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
                                        equity = self.portfolio.equity(current_prices)
                                        pair_allocation = min(
                                            10000, equity * 0.02
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
                                            res_a = self.executor.place_order(order_a)
                                            if res_a.ok:
                                                await self.db.record_trade(
                                                    symbol_a,
                                                    "BUY",
                                                    qty_a,
                                                    res_a.fill_price or price_a,
                                                    0,
                                                )
                                                logger.info(
                                                    f"Pairs trade: Bought {qty_a} {symbol_a} at ${price_a:.2f}"
                                                )

                                            # Short symbol_b (if shorting enabled, otherwise skip)
                                            if self.cfg.execution.enable_short_selling:
                                                order_b = Order(
                                                    symbol=symbol_b,
                                                    quantity=qty_b,
                                                    side="SELL",
                                                    price=price_b,
                                                )
                                                res_b = self.executor.place_order(order_b)
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
                                            res_b = self.executor.place_order(order_b)
                                            if res_b.ok:
                                                await self.db.record_trade(
                                                    symbol_b,
                                                    "BUY",
                                                    qty_b,
                                                    res_b.fill_price or price_b,
                                                    0,
                                                )
                                                logger.info(
                                                    f"Pairs trade: Bought {qty_b} {symbol_b} at ${price_b:.2f}"
                                                )
                                                if hasattr(self, "trades_executed"):
                                                    self.trades_executed += 1

                                            # Short symbol_a (if shorting enabled, otherwise skip)
                                            if self.cfg.execution.enable_short_selling:
                                                order_a = Order(
                                                    symbol=symbol_a,
                                                    quantity=qty_a,
                                                    side="SELL",
                                                    price=price_a,
                                                )
                                                res_a = self.executor.place_order(order_a)
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

            # Disconnect via ConnectionManager if exists
            if hasattr(self, "conn_mgr") and self.conn_mgr:
                logger.info("Disconnecting from IBKR...")
                await self.conn_mgr.disconnect()

            # Legacy IB Gateway disconnect (for backward compatibility)
            if hasattr(self, "ib") and self.ib and self.ib.isConnected():
                logger.info("Disconnecting from IB Gateway...")
                self.ib.disconnect()

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
    duration: str = "10 D",
    bar_size: str = "30 mins",
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
    use_ml_strategy: bool = False,
    use_ml_enhanced: bool = False,
    use_smart_execution: bool = False,
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

    while not shutdown_flag:
        eastern = pytz.timezone("US/Eastern")
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
            logger.info(
                f"Starting trading cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            )

            # Run the trading system with proper cleanup
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
                use_smart_execution=use_smart_execution,
            )

            try:
                await runner.run(symbols)
            finally:
                # Ensure proper cleanup of runner resources
                await runner.cleanup()

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
    args = parser.parse_args()

    cfg = load_config()
    if cfg.execution.mode == "live" and not args.confirm_live:
        raise SystemExit("Refusing to run in live mode without --confirm-live")

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
            )
        )


if __name__ == "__main__":
    main()
