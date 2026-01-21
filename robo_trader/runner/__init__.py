"""
Runner subpackage - modular trading system components.

Architecture:
- data_fetcher.py: Market data retrieval (IBKR historical bars, caching) ✅
- subprocess_manager.py: IBKR subprocess worker lifecycle ✅
- signal_generator.py: Strategy signal generation (ML, technical, news) [TODO]
- trade_executor.py: Order execution and position management [TODO]
- portfolio_tracker.py: Position tracking and P&L calculation [TODO]

Current Status:
Core functionality in runner_async.py, with data_fetcher and subprocess_manager
extracted as standalone modules. These modules can be used independently or
the AsyncRunner will continue to use its internal methods for backwards compatibility.

Migration Plan:
1. ✅ Extract data fetching logic (DataFetcher class)
2. ✅ Extract subprocess management (SubprocessManager class)
3. [ ] Split process_symbol into signal generation + trade execution
4. [ ] Extract portfolio tracking
5. [ ] Refactor AsyncRunner to use extracted modules

Usage:
    # Use extracted modules directly
    from robo_trader.runner import DataFetcher, SubprocessManager

    # Or use the full AsyncRunner (backwards compatible)
    from robo_trader.runner import AsyncRunner
"""

from ..runner_async import AsyncRunner, SymbolResult
from .data_fetcher import DataFetcher
from .subprocess_manager import SubprocessManager

__all__ = [
    "AsyncRunner",
    "SymbolResult",
    "DataFetcher",
    "SubprocessManager",
]
