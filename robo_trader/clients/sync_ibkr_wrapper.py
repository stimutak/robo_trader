"""
DEPRECATED: Use `robo_trader.connection_manager.ConnectionManager` instead.

This module remains only for import compatibility and will be removed in a
future release.
"""

import warnings


class SyncIBKRWrapper:  # noqa: D401
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "sync_ibkr_wrapper is deprecated; use ConnectionManager",
            DeprecationWarning,
            stacklevel=2,
        )
        raise RuntimeError("Deprecated module: use robo_trader.connection_manager")
