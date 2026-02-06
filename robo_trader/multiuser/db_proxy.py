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

from robo_trader.database_async import DEFAULT_PORTFOLIO_ID


# Methods that accept portfolio_id parameter
_PORTFOLIO_SCOPED_METHODS = frozenset({
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
    "upsert_portfolio",
})


class PortfolioScopedDB:
    """Proxy that auto-injects portfolio_id into portfolio-scoped DB methods.

    For global methods (market_data, ticks, features, etc.), calls pass through
    to the underlying database unchanged.
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

        return attr
