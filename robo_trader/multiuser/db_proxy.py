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
        # D-8: cleanup_old_data MUST be portfolio-scoped via the proxy so a
        # scoped holder cannot accidentally blanket-clean across portfolios.
        # The underlying method still cleans truly-global tables (market_data,
        # ticks) unconditionally and only touches the signals table when an
        # explicit portfolio_id is provided. Callers that intentionally want
        # to skip the per-portfolio signal cleanup (e.g. periodic global
        # market_data cleanup) MUST use the underlying ``_db`` reference
        # directly: ``scoped_db._db.cleanup_old_data(...)``.
        "cleanup_old_data",
    }
)

# Methods that are known-global (no portfolio_id) -- skip warnings for these.
# upsert_portfolio takes portfolio_data dict with id inside, not a kwarg.
_KNOWN_GLOBAL_METHODS = frozenset(
    {
        "initialize",
        "close",
        "health_check",
        "ensure_connection",
        "get_connection",
        "store_market_data",
        "batch_store_market_data",
        "get_latest_market_data",
        "get_all_positions",
        "get_portfolios",
        "upsert_portfolio",
    }
)

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

        # Non-callable attributes and dunder/private names pass through unchanged.
        if not callable(attr) or name.startswith("_"):
            return attr

        if name in _PORTFOLIO_SCOPED_METHODS:
            # Wrap to auto-inject portfolio_id
            async def scoped_method(*args, **kwargs):
                if "portfolio_id" not in kwargs:
                    kwargs["portfolio_id"] = self.portfolio_id
                return await attr(*args, **kwargs)

            return scoped_method

        if name in _KNOWN_GLOBAL_METHODS:
            return attr

        # Deny-by-default: any callable not explicitly listed in either set is
        # refused to prevent silent cross-portfolio leaks. Add the method to the
        # appropriate set in db_proxy.py if the call is intentional.
        raise AttributeError(
            f"PortfolioScopedDB refuses to call '{name}': not in "
            f"_PORTFOLIO_SCOPED_METHODS or _KNOWN_GLOBAL_METHODS. "
            f"This prevents silent cross-portfolio leaks. "
            f"Add '{name}' to the appropriate set if intentional."
        )
