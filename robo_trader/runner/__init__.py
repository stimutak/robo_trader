"""
Runner subpackage - modular trading system components.

Architecture:
- data_fetcher.py: Market data retrieval (IBKR historical bars, caching) ✅
- subprocess_manager.py: IBKR subprocess worker lifecycle ✅
- signal_generator.py: Strategy signal generation (ML, technical, news) ✅
- trade_executor.py: Order execution and position management ✅
- portfolio_tracker.py: Position tracking and P&L calculation ✅

Current Status:
All 5 modules have been extracted from runner_async.py. These modules can be
used independently or the AsyncRunner will continue to use its internal methods
for backwards compatibility.

Migration Plan:
1. ✅ Extract data fetching logic (DataFetcher class)
2. ✅ Extract subprocess management (SubprocessManager class)
3. ✅ Extract signal generation (SignalGenerator class)
4. ✅ Extract trade execution (TradeExecutor class)
5. ✅ Extract portfolio tracking (PortfolioTracker class)
6. [ ] Refactor AsyncRunner to use extracted modules (optional future work)

Usage:
    # Use extracted modules directly
    from robo_trader.runner import (
        DataFetcher,
        SubprocessManager,
        SignalGenerator,
        TradeExecutor,
        PortfolioTracker,
    )

    # Or use the full AsyncRunner (backwards compatible)
    from robo_trader.runner import AsyncRunner
"""

from ..runner_async import AsyncRunner, SymbolResult
from .data_fetcher import DataFetcher
from .portfolio_tracker import PortfolioTracker
from .signal_generator import SignalGenerator, SignalResult
from .subprocess_manager import SubprocessManager
from .trade_executor import ExecutionResult, Order, TradeExecutor

__all__ = [
    # Main runner (backwards compatible)
    "AsyncRunner",
    "SymbolResult",
    # Extracted modules
    "DataFetcher",
    "SubprocessManager",
    "SignalGenerator",
    "SignalResult",
    "TradeExecutor",
    "ExecutionResult",
    "Order",
    "PortfolioTracker",
]
