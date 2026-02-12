"""
Portfolio-scoped database proxy.

Wraps AsyncTradingDatabase to automatically inject portfolio_id into all
portfolio-scoped method calls. This avoids having to modify 30+ call sites
in runner_async.py.

Usage:
    db = AsyncTradingDatabase()
    scoped_db = PortfolioScopedDB(db, portfolio_id="aggressive")

    # These automatically scope to the "aggressive" portfolio:
    await scoped_db.get_positions()           # → db.get_positions(portfolio_id="aggressive")
    await scoped_db.record_trade(...)         # → db.record_trade(..., portfolio_id="aggressive")
    await scoped_db.update_account(...)       # → db.update_account(..., portfolio_id="aggressive")

    # Global methods (no portfolio_id) pass through unchanged:
    await scoped_db.store_market_data(...)    # → db.store_market_data(...)
"""

import inspect

from ..logger import get_logger
from robo_trader.database_async import DEFAULT_PORTFOLIO_ID

logger = get_logger(__name__)

# Methods that accept portfolio_id parameter
_PORTFOLIO_SCOPED_METHODS = frozenset(
    {
        "update_position",
        "record_trade",
        "update_account",
        "record_signal",
        "get_position",
        "get_positions",
        "has_recent_buy_trade",
        "has_recent_sell_trade",
        "get_recent_trades",
        "get_account_info",
        "save_equity_snapshot",
        "get_equity_history",
        "portfolio_exists",
    }
)

# Methods that are known-global (no portfolio_id) -- skip warnings for these.
# upsert_portfolio takes portfolio_data dict with id inside, not a kwarg.
_KNOWN_GLOBAL_METHODS = frozenset(
    {
        "initialize",
        "get_connection",
        "store_market_data",
        "get_latest_market_data",
        "get_all_positions",
        "get_portfolios",
        "upsert_portfolio",
    }
)

# Cache for methods we've already warned about (avoid log spam)
_warned_methods: set = set()


class PortfolioScopedDB:
    """Proxy that auto-injects portfolio_id into portfolio-scoped DB methods.

    For global methods (market_data, ticks, features, etc.), calls pass through
    to the underlying database unchanged.

    IMPORTANT: When adding a new method to AsyncTradingDatabase that takes
    portfolio_id, you MUST also add it to _PORTFOLIO_SCOPED_METHODS above.
    Otherwise calls through this proxy will silently skip scoping, leaking
    data across portfolios.
    """

    def __init__(self, db, portfolio_id: str = DEFAULT_PORTFOLIO_ID):
        self._db = db
        self.portfolio_id = portfolio_id

    def __getattr__(self, name):
        attr = getattr(self._db, name)

        if name in _PORTFOLIO_SCOPED_METHODS and callable(attr):
            # Wrap to auto-inject portfolio_id
            async def scoped_method(*args, **kwargs):
                if "portfolio_id" not in kwargs:
                    kwargs["portfolio_id"] = self.portfolio_id
                return await attr(*args, **kwargs)

            return scoped_method

        # Scope-leak detection: warn if a callable has portfolio_id in its
        # signature but isn't in the scoped set (and isn't known-global).
        if callable(attr) and name not in _KNOWN_GLOBAL_METHODS:
            if name not in _warned_methods:
                try:
                    sig = inspect.signature(attr)
                    if "portfolio_id" in sig.parameters:
                        logger.warning(
                            f"SCOPE LEAK: Method '{name}' accepts portfolio_id but is NOT in "
                            f"_PORTFOLIO_SCOPED_METHODS. Calls through PortfolioScopedDB will "
                            f"use the default portfolio, not '{self.portfolio_id}'. "
                            f"Add '{name}' to _PORTFOLIO_SCOPED_METHODS in db_proxy.py."
                        )
                        _warned_methods.add(name)
                except (ValueError, TypeError):
                    pass  # Can't inspect (e.g. built-in) -- skip

        return attr
